import cv2 as cv
import numpy as np

proto = 'deploy.prototxt'
weights = '/home/dkurtaev/Downloads/res10_300x300_ssd_iter_140000.caffemodel'

netFromCaffe = cv.dnn.readNetFromCaffe(proto, weights)
netFromTensorflow = cv.dnn.readNetFromTensorflow('face_detector.pb')

np.random.seed(2701)
inputData = np.random.standard_normal([1, 3, 300, 300]).astype(np.float32)

netFromCaffe.setInput(inputData)
netFromTensorflow.setInput(inputData)

out1 = netFromCaffe.forward(['mbox_loc', 'mbox_conf_flatten', 'detection_out'])
out2 = netFromTensorflow.forward(['mbox_loc', 'mbox_conf_flatten', 'DetectionOutput'])

print 'Max diff @ locations:  %e' % np.max(np.abs(out1[0] - out2[0]))
print 'Max diff @ confidence: %e' % np.max(np.abs(out1[1] - out2[1]))
print 'Max diff @ detections: %e' % np.max(np.abs(out1[2] - out2[2]))
