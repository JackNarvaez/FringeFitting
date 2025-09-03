"""
Plot the sky brightness from a FITS file.
"""

import astropy.io.fits          as fits
import matplotlib.pyplot        as plt
import numpy                    as np
import matplotlib.font_manager  as fm
from matplotlib                                 import rcParams
from matplotlib.ticker                          import ScalarFormatter
from mpl_toolkits.axes_grid1.anchored_artists   import AnchoredSizeBar
from mpl_toolkits.axes_grid1.inset_locator      import inset_axes
from matplotlib.patches                         import Rectangle, Circle

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

########## === Load main FITS === ##########

hdu     = fits.open("../../Data/LOFAR/wsclean-image_05_1024_me.fits")[0]
header  = hdu.header
image   = hdu.data[0, 0, :, :]

# Pixel scale (deg/pix) -> arcsec/pix
cdelt_ra    = header["CDELT1"] * 3600
cdelt_dec   = header["CDELT2"] * 3600
crpix1, crpix2 = header["CRPIX1"], header["CRPIX2"]

ny, nx  = image.shape

# Extent in arcsec relative to center
x = (np.arange(nx) - crpix1) * cdelt_ra
y = (np.arange(ny) - crpix2) * cdelt_dec
extent = [x.min(), x.max(), y.min(), y.max()]

# Plot main image
fig, ax = plt.subplots()
im = ax.imshow(image, cmap="inferno", vmin=0, vmax=None, extent=extent, origin="lower")

#######################################################################
# Calculate S/N of the source w.r.t. the background

X, Y = np.meshgrid(x, y)   # X: RA offsets, Y: Dec offsets
R = np.sqrt((X-0.08)**2 + (Y+0.02)**2)

r_inner = 0.3   # central circle radius
r_ring_inner = 0.3
r_ring_outer = 1.0

# Add circles centered on (0,0) because extent is relative to CRPIX center
circle_inner = Circle((0.08, -0.02), r_inner, color='cyan', fill=False, linewidth=1.5, label='Central circle')
circle_ring_inner = Circle((0.08, -0.02), r_ring_inner, color='lime', linestyle='--', fill=False, linewidth=1.5, label='Ring inner')
circle_ring_outer = Circle((0.08, -0.02), r_ring_outer, color='lime', linestyle='--', fill=False, linewidth=1.5, label='Ring outer')

# Add circles to the axes
ax.add_patch(circle_inner)
ax.add_patch(circle_ring_inner)
ax.add_patch(circle_ring_outer)

mask_central = (R <= r_inner)
mask_ring = (R > r_ring_inner) & (R <= r_ring_outer)

# --- Compute mean and sum intensities ---
central_mean = np.mean(np.abs(image[mask_central]))
central_sum = np.sum(np.abs(image[mask_central]))

ring_mean = np.mean(np.abs(image[mask_ring]))
ring_sum = np.sum(np.abs(image[mask_ring]))

print(f"Central circle (r <= {r_inner} arcsec):")
print(f"  Sum = {central_sum}, Mean = {central_mean}")
print(f"Ring ({r_ring_inner} < r <= {r_ring_outer} arcsec):")
print(f"  Sum = {ring_sum}, Mean = {ring_mean}")
print(f"{central_mean/ring_mean}\t {central_sum/ring_sum}\n")

###########################################################################

ax.set_xlabel("Relative R.A. (arcsec)")
ax.set_ylabel("Relative Decl. (arcsec)")

# Colorbar
cbar = plt.colorbar(im, ax=ax, pad=0.02)
cbar.formatter = ScalarFormatter(useMathText=True)
cbar.formatter.set_powerlimits((-2, 2))
cbar.update_ticks()
cbar.set_label("Flux (Jy/beam)")

# Add scalebar (5 arcsec) on main image
scalebar = AnchoredSizeBar(ax.transData,
                           5.0, "5\"", "lower right",
                           pad=0.5, color="white", frameon=False,
                           size_vertical=0.1,
                           fontproperties=fm.FontProperties(size=14))
ax.add_artist(scalebar)

########## === Load second FITS for zoom === ##########

hdu1    = fits.open("../../Data/LOFAR/wsclean-image_005_1024_me.fits")[0]
image1  = hdu1.data[0, 0, :, :]

# Define zoom region (in pixels of main image)
center_y, center_x = image.shape[0] // 2, image.shape[1] // 2
zoom_size = min(image.shape) // 20  # Adjust zoom level as needed
y_start = max(0, center_y - zoom_size)
y_end   = min(image.shape[0], center_y + zoom_size)
x_start = max(0, center_x - zoom_size)
x_end   = min(image.shape[1], center_x + zoom_size)

# Rectangle coordinates need to be in arcsec
x0_arc  = (x_start - crpix1) * cdelt_ra
y0_arc  = (y_start - crpix2) * cdelt_dec
width_arc   = (x_end - x_start) * cdelt_ra
height_arc  = (y_end - y_start) * cdelt_dec

# Add rectangle on main image
zoom_rect = Rectangle((x0_arc, y0_arc), width_arc, height_arc,
                     linewidth=1.5, edgecolor='white', facecolor='none',
                     linestyle='--', alpha=0.9)
ax.add_patch(zoom_rect)

# Create inset axes (upper right corner)
inset_ax = inset_axes(ax, width="35%", height="35%", loc="upper right")

# Plot zoomed image in inset with proper scale
inset_ax.imshow(image1, cmap="inferno", vmin=0, origin="lower", extent=[x0_arc, x0_arc + width_arc,
                        y0_arc, y0_arc + height_arc])

# Remove inset ticks
inset_ax.set_xticks([])
inset_ax.set_yticks([])

# Add white border to inset
for spine in inset_ax.spines.values():
    spine.set_color("white")
    spine.set_linewidth(1)

# Add scalebar inside inset (example: 1 arcsec)
inset_scalebar = AnchoredSizeBar(inset_ax.transData,
                                 1.0, "1\"", "lower right",
                                 pad=0.3, color="white", frameon=False,
                                 size_vertical=0.02,
                                 fontproperties=fm.FontProperties(size=10))
inset_ax.add_artist(inset_scalebar)

# Remove inset ticks
inset_ax.set_xticks([])
inset_ax.set_yticks([])

# Add white border to inset
for spine in inset_ax.spines.values():
    spine.set_color("white")
    spine.set_linewidth(1)

plt.savefig("Im.png", dpi=400, transparent=True)
plt.show()