import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

######################################
### SEBlock ##########################
######################################

######################################
### High Resolution:
img = Image.open("/Users/Florian-Kuhm/Documents/bee_images/bee_000003/sequence_000002/image_000082.png")
img = img.crop(box = (img.width//2-img.width//3, img.height//2-img.height//3-30,img.width//2+img.width//3, img.height//2+img.height//3))
img = img.resize(size = (256,256))

# Save image to file:
img.save("/Users/Florian-Kuhm/Desktop/imgExample1.png")

######################################
### Low Resolution:
img = Image.open("/Users/Florian-Kuhm/Documents/bee_images/bee_000003/sequence_000002/image_000082.png")
img = img.crop(box = (img.width//2-img.width//3, img.height//2-img.height//3-30,img.width//2+img.width//3, img.height//2+img.height//3))
img = img.resize(size = (16,16))

# Save image to file:
img.save("/Users/Florian-Kuhm/Desktop/imgExample2.png")

plt.figure()
plt.imshow(img)
plt.axis('off')
plt.show()


######################################
### Multiply channels with weight:
img = Image.open("/Users/Florian-Kuhm/Documents/bee_images/bee_000003/sequence_000002/image_000082.png")
img = img.crop(box = (img.width//2-img.width//3, img.height//2-img.height//3-30,img.width//2+img.width//3, img.height//2+img.height//3))
img = img.resize(size = (256,256))

np.random.seed(123)
r, g, b = img.split()
r = r.point(lambda i: i * 0.7 * np.random.rand())
g = g.point(lambda i: i * 0.1 * np.random.rand())
b = b.point(lambda i: i * 0.2 * np.random.rand())

img = Image.merge('RGB', (r, g, b))

plt.imshow(img)

img.save("/Users/Florian-Kuhm/Desktop/imgExample3.png")

######################################
### Discriminator: ###################
######################################



