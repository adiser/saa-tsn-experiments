import cv2
import pafy
import numpy as np
from openpose import OpenPose 
import caffe

class Param:
    defRes = 360
    scales = [1, 0.5]
    caffemodel = "../models/pose/body_25/pose_iter_584000.caffemodel"
    prototxt = "../models/pose/body_25/pose_deploy.prototxt"

# def load_openpose():
#     params = dict()

#     params["logging_level"] = 3
#     params["output_resolution"] = "-1x-1"
#     params["net_resolution"] = "-1x"+str(Param.defRes)
#     params["model_pose"] = "BODY_25"
#     params["alpha_pose"] = 0.6
#     params["scale_gap"] = 0.5
#     params["scale_number"] = len(Param.scales)
#     params["render_threshold"] = 0.05
#     params["num_gpu_start"] = 0
#     params["disable_blending"] = False
#     params["default_model_folder"] = "../models"

#     openpose = OpenPose(params)

#     return openpose

def load_caffenets():
    caffe.set_mode_gpu()
    caffe.set_device(0)
    nets = []
    for scale in Param.scales:
        nets.append(caffe.Net(Param.prototxt, Param.caffemodel, caffe.TEST))
    print("Net loaded")
    
    return nets

first_run = True

def generate_pose_heatmap(frame, nets):

    imagesForNet, imagesOrig = OpenPose.process_frames(frame, Param.defRes, Param.scales)

    global first_run
    if first_run:
        for i in range(0, len(Param.scales)):
            net = nets[i]
            imageForNet = imagesForNet[i]
            in_shape = net.blobs['image'].data.shape
            in_shape = (1, 3, imageForNet.shape[1], imageForNet.shape[2])
            net.blobs['image'].reshape(*in_shape)
            net.reshape()

        first_run = False
        print("Reshaped")

    heatmaps = []
    for i in range(0, len(Param.scales)):
        net = nets[i]
        imageForNet = imagesForNet[i]
        net.blobs['image'].data[0,:,:,:] = imageForNet
        net.forward()
        heatmaps.append(net.blobs['net_output'].data[:,:,:,:])

    # array, frame = openpose.poseFromHM(frame, heatmaps, Param.scales)

    # hm = heatmaps[0][:,0:18,:,:]; frame = OpenPose.draw_all(imagesOrig[0], hm, -1, 1, True)
    paf = heatmaps[0][:,20:,:,:]; frame = OpenPose.draw_all(imagesOrig[0], paf, -1, 4, False)

    return frame

class Queue():
    def __init__(self, size):
        self.size = size
        self.container = []
    
    def enqueue(self, item):
        if len(self.container) < self.size:
            self.container.append(item)
        else:
            print('Buffer full')

    def dequeue(self):
        if not self.isempty():
            self.container.pop(0)
        else:
            print("Buffer empty")

    def get(self):
        return np.array(self.container)

    def isempty(self):
        return len(self.container) == 0

    def isfull(self):
        return (len(self.container) == self.size)


def arp(imgs):
    """
    Exact replica of the rank pooling algorithm described in the paper, 
    including the equations and notations used
    args:
        imgs : stack of rgb images 
    """

    T = len(imgs)
  
    harmonics = []
    harmonic = 0
    for t in range(0, T+1):
        harmonics.append(harmonic)
        harmonic += float(1)/(t+1)

    weights = []
    for t in range(1,T+1):
        weight = 2 * (T - t + 1) - (T+1) * (harmonics[T] - harmonics[t-1])
        weights.append(weight)
        
    feature_vectors = []
    for i in range(len(weights)):
        feature_vectors.append(imgs[i] * weights[i])

    feature_vectors = np.array(feature_vectors)
    
    rank_pooled = np.sum(feature_vectors, 0)
    rank_pooled = cv2.normalize(rank_pooled, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

    return rank_pooled
    
def main():
    nets = load_caffenets()

    url = 'https://www.youtube.com/watch?v=BaYeMPgg5sY'
    vpafy = pafy.new(url)
    play = vpafy.getbest(preftype = 'webm')
    cap = cv2.VideoCapture(play.url)

    buffer = Queue(10)

    while True:
        
        ret, frame = cap.read()

        if not buffer.isfull():
            buffer.enqueue(frame)
        else:
            buffer.dequeue()
            buffer.enqueue(frame)

        pose_heatmap = generate_pose_heatmap(frame, nets)
        pose_heatmap = cv2.normalize(pose_heatmap, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

        # frames = buffer.get()
        # rank_pooled = arp(frames)

        # final_frame = cv2.addWeighted(rank_pooled, 0.5, pose_heatmap, 0.5, 0.5)
        cv2.imshow('frame', pose_heatmap)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()