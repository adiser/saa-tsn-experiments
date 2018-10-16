OPENPOSE_URL="https://www.posefs1.perception.cs.cmu.edu/OpenPose/models/"
POSE_FOLDER="pose/"
BODY_25_FOLDER="body_25"
BODY_25_MODEL=${BODY_25_FOLDER}"pose_iter_584000.caffemodel"

wget -c ${OPENPOSE_URL}${BODY_25_MODEL} -P ${BODY_25_FOLDER} 

