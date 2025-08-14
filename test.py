import astropy.io.fits as fits
import astropy.wcs as wcs_lib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import ScalarFormatter
from astropy.visualization.wcsaxes import add_scalebar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from astropy.visualization.wcsaxes import WCSAxes

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
hdu = fits.open("../Data/028/wsclean-image_BL_05_1024.fits")[0]
wcs = wcs_lib.WCS(hdu.header)

wcs_2d = wcs.celestial

image = hdu.data[0, 0, :, :]

hdu1 = fits.open("../Data/028/wsclean-image_BL_005_1024.fits")[0]
wcs1 = wcs_lib.WCS(hdu1.header)
wcs1_2d = wcs1.celestial
image1 = hdu1.data[0, 0, :, :]

fig = plt.figure()
ax = plt.subplot(projection=wcs_2d)

im = ax.imshow(image, cmap="inferno", origin="lower", vmin=0., vmax=0.35)

# Hide ticks
# ax.coords[0].set_ticklabel_visible(False)
ax.coords[1].set_ticklabel_visible(False)

# Colorbar
cbar = plt.colorbar(im, ax=ax, pad=0.02)
cbar.formatter = ScalarFormatter(useMathText=True)
cbar.formatter.set_powerlimits((-2, 2))
cbar.set_label("Flux(Jy/beam)")
cbar.update_ticks()



scalebar = add_scalebar(ax, 5/3600, label="5 \'", color="white")

ax.grid(False)
ax.set_xlabel("Right ascension")

# Create inset subplot with WCS projection
inset_ax = inset_axes(ax, width="40%", height="40%", loc='upper right', 
                      bbox_to_anchor=(0.01, 0.01, 1, 1),
                      bbox_transform=ax.transAxes,
                      axes_class=WCSAxes,
                      axes_kwargs=dict(wcs=wcs1_2d))

# Get center coordinates and zoom region
center_y, center_x = image.shape[0] // 2, image.shape[1] // 2
zoom_size = min(image.shape) // 20  # Adjust zoom level as needed

# Extract zoom region
y_start = max(0, center_y - zoom_size)
y_end = min(image.shape[0], center_y + zoom_size)
x_start = max(0, center_x - zoom_size)
x_end = min(image.shape[1], center_x + zoom_size)

zoom_image = image[y_start:y_end, x_start:x_end]

# Display zoomed image in inset
inset_ax.imshow(image1, cmap="inferno", vmin=0, vmax=0.35, origin="lower")

# Remove ticks and labels from inset
inset_ax.coords[0].set_ticklabel_visible(False)
inset_ax.coords[1].set_ticklabel_visible(False)

scalebar = add_scalebar(inset_ax, 1/3600, label="1 \'", color="white")

# Add a border to the inset
for spine in inset_ax.spines.values():
    spine.set_color('white')
    spine.set_linewidth(1)

inset_ax.grid(False)

# Optional: Add a rectangle on the main plot to show zoom region
from matplotlib.patches import Rectangle
zoom_rect = Rectangle((x_start, y_start), x_end - x_start, y_end - y_start,
                     linewidth=1, edgecolor='white', facecolor='none', 
                     linestyle='--', alpha=0.8)
ax.add_patch(zoom_rect)

plt.savefig("Imagescale.png", dpi=300, transparent=True)
plt.show()