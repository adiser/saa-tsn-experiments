cd data
wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar
unrar x hmdb51-org.rar rars/
for a in $(ls rars); do unrar x "rars/${a}" videos/; done;

cd ../utils/temporal-segment-networks
bash build_all.sh
bash scripts/extract_optical_flow.sh ../../data/hmdb51 ../../data/hmdb51_frames 4

cd ../envs
python3 -m venv pytorch
source pytorch/bin/activate
pip install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp35-cp35m-linux_x86_64.whl 
pip install numpy
pip install torchvision


