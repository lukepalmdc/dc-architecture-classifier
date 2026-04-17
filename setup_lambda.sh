#!/bin/bash
# Setup script for Lambda Labs GPU instance
# Run once after cloning the repo: bash setup_lambda.sh

set -e

echo "=== Installing dependencies ==="
pip install -r requirements.txt

echo ""
echo "=== Verifying GPU ==="
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo ""
echo "=== Data setup ==="
echo "You need to transfer data files to this instance."
echo ""
echo "From your local machine, run:"
echo "  rsync -avz --progress data/styles/     ubuntu@<LAMBDA_IP>:~/dc_architecture/data/styles/"
echo "  rsync -avz --progress data/images/     ubuntu@<LAMBDA_IP>:~/dc_architecture/data/images/"
echo "  scp data/buildings_enriched.geojson    ubuntu@<LAMBDA_IP>:~/dc_architecture/data/"
echo ""
echo "Or if you have them on S3:"
echo "  aws s3 sync s3://your-bucket/dc_architecture/data/ data/"
echo ""
echo "=== Ready to train ==="
echo "Single run:"
echo "  python train_architecture.py --name my_run"
echo ""
echo "All experiments:"
echo "  python run_experiments.py"
