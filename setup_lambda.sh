#!/bin/bash
# Setup script for Lambda Labs or RunPod GPU instance
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
echo "Lambda Labs (port 22, user ubuntu):"
echo "  rsync -avz --progress data/styles/          ubuntu@<IP>:~/dc_architecture/data/styles/"
echo "  rsync -avz --progress data/images/          ubuntu@<IP>:~/dc_architecture/data/images/"
echo "  scp data/buildings_enriched.geojson         ubuntu@<IP>:~/dc_architecture/data/"
echo ""
echo "RunPod (custom port, user root — get IP+port from pod page):"
echo "  rsync -avz -e 'ssh -p <PORT>' --progress data/styles/   root@<IP>:~/dc_architecture/data/styles/"
echo "  rsync -avz -e 'ssh -p <PORT>' --progress data/images/   root@<IP>:~/dc_architecture/data/images/"
echo "  scp -P <PORT> data/buildings_enriched.geojson            root@<IP>:~/dc_architecture/data/"
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
