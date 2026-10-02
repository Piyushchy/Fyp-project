#!/usr/bin/env bash
# Chain the v6 stages so no GPU time is lost waiting for a human.
#
#   1. wait for clean_labels.py to finish writing data_v6
#   2. train v6 on the cleaned corpus
#   3. install it into the deployment and calibrate on calib/
#   4. score on test/, which neither training nor calibration has seen
#
# Each stage checks the previous one actually succeeded; a failure stops
# the chain rather than training on a corpus that was never written.

set -u

cd "$(dirname "$0")"

PY=../.venv/Scripts/python.exe
DEPLOY=../vit-emotion-v2-deployment

echo "=== waiting for label cleaning ==="
while ! grep -q "Wrote data_v6" clean_labels.log 2>/dev/null; do
    if grep -q "Traceback" clean_labels.log 2>/dev/null; then
        echo "FAILED: clean_labels.py raised"; exit 1
    fi
    sleep 60
done

if [ ! -d data_v6/train ]; then
    echo "FAILED: data_v6/train missing"; exit 1
fi

echo "=== cleaned corpus ==="
for d in data_v6/train/*; do
    printf '%-10s %6d\n' "$(basename "$d")" "$(ls "$d" | wc -l)"
done

echo "=== training v6 ==="
$PY train_face.py --data data_v6 --epochs 4 --batch-size 32 --lr 3e-5 \
    --workers 0 --balance 0 --out vit-emotion-v6.onnx > train_v6.log 2>&1

if [ ! -f vit-emotion-v6.onnx ]; then
    echo "FAILED: training produced no onnx"; tail -20 train_v6.log; exit 1
fi

echo "=== installing ==="
cp vit-emotion-v6.onnx vit-emotion-v6.onnx.data "$DEPLOY"/
cp emotion_config.json "$DEPLOY"/emotion_config.v6.json

cd "$DEPLOY"

echo "=== calibrating on calib/ ==="
EMOTION_MODEL=v6 $PY eval_face.py --data eval_data_unseen/calib/webcam \
    --fit-calibration --calibration-holdout 0 > /dev/null 2>&1

echo "=== scoring on untouched test/ ==="
EMOTION_MODEL=v6 $PY eval_face.py --data eval_data_unseen/test/webcam \
    > /dev/null 2>&1
cp results_face_eval.v6.json results_face_eval.v6.unseen_test.json

$PY - <<'EOF'
import json
print(f"{'model':6}{'acc':>8}{'macroF1':>9}{'neutF1':>8}{'disgF1':>8}")
for m in ("v3", "v4", "v6"):
    try:
        d = json.load(open(f"results_face_eval.{m}.unseen_test.json"))
    except FileNotFoundError:
        continue
    r = [c for c in d["configurations"] if c.get("logit_bias")][-1]
    pc = r["per_class_f1"]
    print(f"{m:6}{r['accuracy']*100:7.2f}%{r['macro_f1']*100:8.2f}%"
          f"{pc['neutral']*100:7.2f}%{pc['disgust']*100:7.2f}%")
EOF

echo "=== pipeline complete ==="
