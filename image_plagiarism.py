#!/usr/bin/env python
# coding: utf-8

# In[1]:


from skimage.metrics import structural_similarity
import cv2
import numpy as np
from IPython.display import Image


# In[2]:


def orb_sim(img1, img2):
  # SIFT is no longer available in cv2 so using ORB
  orb = cv2.ORB_create()

  # detect keypoints and descriptors
  kp_a, desc_a = orb.detectAndCompute(img1, None)
  kp_b, desc_b = orb.detectAndCompute(img2, None)

  # define the bruteforce matcher object
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
  #perform matches. 
  matches = bf.match(desc_a, desc_b)
  #Look for similar regions with distance < 50. Goes from 0 to 100 so pick a number between.
  similar_regions = [i for i in matches if i.distance < 50]  
  if len(matches) == 0:
    return 0
  return len(similar_regions) / len(matches)


# In[3]:


def structural_sim(img1, img2):
  sim,diff = structural_similarity(img1, img2, full=True)
  return sim


# In[4]:


img1 = cv2.imread("BSE_Google.jpg", 0)  
img2 = cv2.imread('BSE_Google.jpg', 0)


# In[5]:


Image("BSE_Google.jpg")


# In[6]:




orb_similarity = orb_sim(img1, img2)  #1.0 means identical. Lower = not similar

print("Similarity using ORB is: ", orb_similarity)


ssim = structural_sim(img1, img2) #1.0 means identical. Lower = not similar
print("Similarity using SSIM is: ", ssim)


# In[7]:


img3 = cv2.imread('BSE_Google_blurred.jpg', 0)


# In[8]:


Image("BSE_Google_blurred.jpg")


# In[9]:



orb_similarity = orb_sim(img1, img3)  #1.0 means identical. Lower = not similar

print("Similarity using ORB is: ", orb_similarity)


ssim = structural_sim(img1, img3) #1.0 means identical. Lower = not similar
print("Similarity using SSIM is: ", ssim)


# In[10]:


img4 = cv2.imread('BSE_Google_noisy.jpg', 0)


# In[11]:


Image("BSE_Google_noisy.jpg")


# In[12]:


orb_similarity = orb_sim(img1, img4)  #1.0 means identical. Lower = not similar

print("Similarity using ORB is: ", orb_similarity)


ssim = structural_sim(img1, img4) #1.0 means identical. Lower = not similar
print("Similarity using SSIM is: ", ssim)


# In[13]:


img5 = cv2.imread('BSE_salt_pepper.jpg', 0)


# In[14]:


Image("BSE_salt_pepper.jpg")


# In[15]:


orb_similarity = orb_sim(img1, img5)  #1.0 means identical. Lower = not similar

print("Similarity using ORB is: ", orb_similarity)


ssim = structural_sim(img1, img5) #1.0 means identical. Lower = not similar
print("Similarity using SSIM is: ", ssim)


# In[16]:


img6 = cv2.imread('monkey.jpg', 0)


# In[17]:


Image("monkey.jpg")


# In[18]:


img7 = cv2.imread('monkey_distorted.jpg', 0)


# In[19]:


Image("monkey_distorted.jpg")


# In[20]:


orb_similarity = orb_sim(img6, img7)  #1.0 means identical. Lower = not similar

print("Similarity using ORB is: ", orb_similarity)


ssim = structural_sim(img6, img7) #1.0 means identical. Lower = not similar
print("Similarity using SSIM is: ", ssim)


# In[ ]:


img1=cv2.imread("BSE_Google.jpg")

img2= cv2.imread("BSE_Google_blurred.jpg")

img2=cv2.resize(img2, (img1.shape[1], img1.shape[0]))

g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

(score, diff) = structural_similarity(g1, g2, full=True)
diff =(diff*255).astype("uint8")

_, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY_INV)

contors = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) [0]

contors = [c for c in contors if cv2.contourArea(c) > 80]

if len(contors):
    for c in contors:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(img1, (x,y), (x+w, y+h), (0,0,255), 4)
while True:
    cv2.imshow("window1", img1)
    cv2.imshow("window2", img2)
    cv2.imshow("window3", diff)
    cv2.imshow("window3", thresh)
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()
        break
print(diff)


# In[ ]:




