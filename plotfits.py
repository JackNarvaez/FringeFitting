import matplotlib.pyplot as plt
import numpy             as np
import astropy.io.fits   as fits
import astropy.wcs       as wcs_lib

hdu = fits.open("./wsclean-image.fits")[0]
wcs = wcs_lib.WCS(hdu.header)
image = hdu.data[0, 0, :, :]
plt.subplot(projection=wcs[0, 0, :, :])
plt.imshow(image)
plt.xlabel("Right ascension")
plt.ylabel("Declination")
plt.title("WSCLEAN image")
plt.colorbar()
plt.show()

hdu1 = fits.open("./wsclean-dirty.fits")[0]
wcs1 = wcs_lib.WCS(hdu1.header)
image1 = hdu1.data[0, 0, :, :]
plt.subplot(projection=wcs1[0, 0, :, :])
plt.imshow(image1)
plt.xlabel("Right ascension")
plt.ylabel("Declination")
plt.title("WSCLEAN dirty image")
plt.colorbar()
plt.show()

hdu1 = fits.open("./corrected-image.fits")[0]
wcs1 = wcs_lib.WCS(hdu1.header)
image1 = hdu1.data[0, 0, :, :]
plt.subplot(projection=wcs1[0, 0, :, :])
plt.imshow(image1)
plt.xlabel("Right ascension")
plt.ylabel("Declination")
plt.title("WSCLEAN corrected image")
plt.colorbar()
plt.show()

plt.subplot(projection=wcs1[0, 0, :, :])
plt.imshow(image==image1, cmap="binary")
plt.xlabel("Right ascension")
plt.ylabel("Declination")
plt.title("Ratio")
plt.colorbar()
plt.show()


print(np.sum(image!=image1))