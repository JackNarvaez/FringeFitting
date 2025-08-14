import astropy.io.fits as fits
import astropy.wcs as wcs_lib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import ScalarFormatter
from astropy.visualization.wcsaxes import add_scalebar

plt.style.use('dark_background')

rcParams.update({
    "font.size": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "lines.markersize": 7,
    "lines.linewidth": 4,
    "figure.figsize": (12, 8),
    "xtick.top": True,
    "xtick.bottom": True,
    "ytick.left": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.direction": 'in',
    "ytick.direction": 'in'
})

# Load FITS
hdu = fits.open("../Data/028/wsclean-image_DATA_05_1024.fits")[0]
wcs = wcs_lib.WCS(hdu.header)

wcs_2d = wcs.celestial

image = hdu.data[0, 0, :, :]

fig = plt.figure()
ax = plt.subplot(projection=wcs_2d)

im = ax.imshow(image, cmap="inferno", vmin=0, vmax=9.5e-3)

# Hide ticks
# ax.coords[0].set_ticks_visible(False)
# ax.coords[1].set_ticks_visible(False)
# ax.coords[0].set_ticklabel_visible(False)
# ax.coords[1].set_ticklabel_visible(False)
ax.set_xlabel("Right ascension")
ax.set_ylabel("Declination")

# Colorbar
cbar = plt.colorbar(im, ax=ax, pad=0.02)
cbar.formatter = ScalarFormatter(useMathText=True)
cbar.formatter.set_powerlimits((-2, 2))
cbar.update_ticks()

# cbar.set_label("Flux(Jy/beam)")
# cbar.update_ticks()

scalebar = add_scalebar(ax, 5/3600, label="5 \'", color="white")

ax.grid(False)
plt.savefig("Imagescale.png", dpi=300, transparent=True)
plt.show()
