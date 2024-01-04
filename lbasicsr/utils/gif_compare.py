import cv2  
import imageio

def img2gif(lq_path, hq_path, save_path, duration=0.75, num_frames=8, num_extra_frames=2):
    '''
    Save a series of images as a GIF.  [LQ, ..., LQ->HQ, ..., HQ]

    Arguments:
        lq_path (str): Path for low-quality (LQ) images.
        hq_path (str): Path for high-quality (HQ) images.
        save_path (str): Path for output GIF.
        duration (float): Duration of each frame in GIF.
        num_frames (int): Number of frames for mixing HQ with LQ images.
        num_extra_frames (int):  Number of frames for single HQ or LQ image.
    '''
    lq = cv2.imread(lq_path)
    hq = cv2.imread(hq_path)
    h,w = hq.shape[:2]

    step = w//num_frames

    images = list()
    for i in range(num_extra_frames):
        images.append(cv2.cvtColor(lq, cv2.COLOR_BGR2RGB))

    for i in range(step, w, step):
        img = cv2.hconcat([hq[:,:i], lq[:,i:]])
        cv2.line(img, (i,0), (i,h), color=(255,255,255), thickness=w//100)
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    for i in range(num_extra_frames):
        images.append(cv2.cvtColor(hq, cv2.COLOR_BGR2RGB))

    imageio.mimsave(save_path, images, "GIF", duration=duration)
