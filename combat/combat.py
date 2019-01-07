#################################################################################
# This module provides some conveniency functions to use in combat notebook     #
#################################################################################
import Tigger
import numpy as np
from astLib.astWCS import WCS
from astropy.io import fits as fitsio
from Tigger.Coordinates import angular_dist_pos_angle

MEGA_HERTZ = 1000000.0


def rad2arcsec(x):
    """Converts `x` from radians to arcseconds"""
    return float(x)*3600.0*180.0/np.pi


def arcsec2deg(x):
    """Converts 'x' from arcseconds to degrees"""
    return float(x)/3600.00


def deg2arcsec(x):
    """Converts 'x' from degrees to arcseconds"""
    return float(x)*3600.00


def update(dict1, dict2):
    """Update dict2 with with dict1 without changing original dict2"""
    tmp = dict(dict2)
    tmp.update(dict1)
    return tmp


def source_position_tolerance(restored_image):
    """Determines the position tolerance of a source using the clean beam size"""
    with fitsio.open(restored_image) as hdu:
        hdr = hdu[0].header
        BMAJ = hdr["BMAJ"]
        BMIN = hdr["BMIN"]
    tolerance_deg = (2*BMAJ + BMIN)/3
    tolerance_rad = (tolerance_deg*np.pi)/180
    return tolerance_rad


def axis_min_max(data_points, error=None, tolerance=0.0):
    """Get max and min from a data set for ploting limit"""
    data_min_max = [1000000.0, -1000000.0]
    err_min_max = [1000000.0, -1000000.0]
    if not error:
        error_points = data_points
    else:
        error_points = error
    for i_in_out, i_out_err in zip(data_points, error_points):
        if i_in_out < data_min_max[0]:
            data_min_max[0] = i_in_out
        if i_in_out > data_min_max[1]:
            data_min_max[1] = i_in_out
        try:
            if i_out_err < err_min_max[0]:
                err_min_max[0] = i_out_err
            if i_out_err > err_min_max[1]:
                err_min_max[1] = i_out_err
        except TypeError:
            pass
    if error:
        return [data_min_max[0]-err_min_max[1]-tolerance,
                data_min_max[1]+err_min_max[1]+tolerance]
    else:
        return [data_min_max[0]-tolerance,
                data_min_max[1]+tolerance]


def fitsInfo(fitsname=None):
    """Get fits header info"""
    hdu = fitsio.open(fitsname)
    hdr = hdu[0].header
    ra = hdr['CRVAL1']
    dra = abs(hdr['CDELT1'])
    raPix = hdr['CRPIX1']
    dec = hdr['CRVAL2']
    ddec = abs(hdr['CDELT2'])
    decPix = hdr['CRPIX2']
    wcs = WCS(hdr, mode='pyfits')
    beam_size = (hdr['BMAJ'], hdr['BMIN'], hdr['BPA'])
    return {'wcs': wcs, 'ra': ra, 'dec': dec,
            'dra': dra, 'ddec': ddec, 'raPix': raPix,
            'decPix': decPix, "beam_size": beam_size}


def get_src_scale(source_shape):
    """Get scale measure of the source in arcsec"""
    if source_shape:
        shape_out = source_shape.getShape()
        shape_out_err = source_shape.getShapeErr()
        minx = shape_out[0]
        majx = shape_out[1]
        minx_err = shape_out_err[0]
        majx_err = shape_out_err[1]
        if minx > 0 and majx > 0:
            scale_out = np.sqrt(minx*majx)
            scale_out_err = np.sqrt(minx_err*minx_err + majx_err*majx_err)
        elif minx > 0:
            scale_out = minx
            scale_out_err = minx_err
        elif majx > 0:
            scale_out = majx
            scale_out_err = majx_err
        else:
            scale_out = 0
            scale_out_err = 0
    else:
        scale_out = 0
        scale_out_err = 0
    scale_out_arc_sec = rad2arcsec(scale_out)
    scale_out_err_arc_sec = rad2arcsec(scale_out_err)
    return scale_out_arc_sec, scale_out_err_arc_sec


def property_results(models, tolerance=0.0001, input='input'):
    results = dict()
    INPUT = input
    for input_model, output_model in models.items():
        heading = output_model[:-9]
        results[heading] = {'models': [input_model, output_model]}
        results[heading]['flux'] = []
        results[heading]['shape'] = []
        results[heading]['position'] = []
        results[heading]['spi'] = []
        props = get_detected_sources_properties(
                '{:s}/{:s}'.format(INPUT, input_model),
                '{:s}/{:s}'.format(INPUT, output_model),
                tolerance)  # TODO area to be same as beam
        for i in range(len(props[0])):
            results[heading]['flux'].append(props[0].items()[i][-1])
        for i in range(len(props[1])):
            results[heading]['shape'].append(props[1].items()[i][-1])
        for i in range(len(props[2])):
            results[heading]['position'].append(props[2].items()[i][-1])
        for i in range(len(props[3])):
            results[heading]['spi'].append(props[3].items()[i][-1])
    return results


def get_detected_sources_properties(model_lsm_file, pybdsm_lsm_file, area_factor):
    """Extracts the output simulation sources properties"""
    model_lsm = Tigger.load(model_lsm_file)
    pybdsm_lsm = Tigger.load(pybdsm_lsm_file)
    # Sources from the input model
    model_sources = model_lsm.sources
    # {"source_name": [I_out, I_out_err, I_in, source_name]}
    targets_flux = dict()       # recovered sources flux
    # {"source_name": [delta_pos_angle_arc_sec, ra_offset, dec_offset,
    #                  delta_phase_centre_arc_sec, I_in]
    targets_position = dict()   # recovered sources position
    # {"source_name: [(majx_out, minx_out, pos_angle_out),
    #                 (majx_in, min_in, pos_angle_in),
    #                 scale_out, scale_out_err, I_in]
    targets_scale = dict()         # recovered sources scale
    targets_spi = dict()
    for model_source in model_sources:
        I_out = 0.0
        I_out_err = 0.0
        name = model_source.name
        RA = model_source.pos.ra
        DEC = model_source.pos.dec
        I_in = model_source.flux.I
        spi_in = model_source.spectrum.spi
        sources = pybdsm_lsm.getSourcesNear(RA, DEC, area_factor)
        # More than one source detected, thus we sum up all the detected sources
        # within a radius equal to the beam size in radians around the true target
        # coordinate
        I_out_err_list = []
        I_out_list = []
        for target in sources:
            I_out_list.append(target.flux.I)
            I_out_err_list.append(target.flux.I_err*target.flux.I_err)
        I_out = sum([val/err for val, err in zip(I_out_list, I_out_err_list)])
        if I_out != 0.0:
            I_out_err = sum([1/I_out_error for I_out_error
                            in I_out_err_list])
            I_out_var_err = np.sqrt(1/I_out_err)
            I_out = I_out/I_out_err
            I_out_err = I_out_var_err
            source = sources[0]
            RA0 = pybdsm_lsm.ra0
            DEC0 = pybdsm_lsm.dec0
            ra = source.pos.ra
            dec = source.pos.dec
            source_name = source.name
            targets_flux[name] = [I_out, I_out_err, I_in, source_name]
            if ra > np.pi:
                ra -= 2.0*np.pi
            delta_pos_angle = angular_dist_pos_angle(RA, DEC, ra, dec)
            delta_pos_angle_arc_sec = rad2arcsec(delta_pos_angle[0])
            delta_phase_centre = angular_dist_pos_angle(RA0, DEC0, ra, dec)
            delta_phase_centre_arc_sec = rad2arcsec(delta_phase_centre[0])
            targets_position[name] = [delta_pos_angle_arc_sec,
                                      rad2arcsec(abs(ra - RA)),
                                      rad2arcsec(abs(dec - DEC)),
                                      delta_phase_centre_arc_sec, I_in,
                                      source_name]
            if source.spectrum:
                spi_out = source.spectrum.spi
                spi_out_err = source.getTags()[0][-1]
                targets_spi[name] = [spi_out, spi_out_err, spi_in,
                                     delta_phase_centre_arc_sec, I_out,
                                     source_name]

            try:
                shape_in = tuple(map(rad2arcsec, model_source.shape.getShape()))
            except AttributeError:
                shape_in = (0, 0, 0)
            if source.shape:
                shape_out = tuple(map(rad2arcsec, source.shape.getShape()))
                shape_out_err = tuple(map(rad2arcsec, source.shape.getShapeErr()))
            else:
                shape_out = (0, 0, 0)
                shape_out_err = (0, 0, 0)
            src_scale = get_src_scale(source.shape)
            targets_scale[name] = [shape_out, shape_out_err, shape_in,
                                   src_scale[0], src_scale[1], I_in,
                                   source_name]
    print("Number of sources recovered: {:d}".format(len(targets_scale)))
    return targets_flux, targets_scale, targets_position, targets_spi


def get_ra_dec_range(area=1.0, phase_centre="J2000,0deg,-30deg"):
    """Get RA and DEC range from area of observations and phase centre"""
    ra = float(phase_centre.split(',')[1].split('deg')[0])
    dec = float(phase_centre.split(',')[2].split('deg')[0])
    d_ra = np.sqrt(area)/2
    d_dec = np.sqrt(area)/2
    ra_range = [ra-d_ra, ra+d_ra]
    dec_range = [dec-d_dec, dec+d_dec]
    return ra_range, dec_range


def noise_sigma(noise_image):
    """Determines the noise sigma level in a dirty image with no source"""
    # Read the simulated noise image
    dirty_noise_hdu = fitsio.open(noise_image)
    # Get the header data unit for the simulated noise
    dirty_noise_data = dirty_noise_hdu[0].data
    # Get the noise sigma
    dirty_noise_data_std = dirty_noise_data.std()
    return dirty_noise_data_std


def source_noise_ratio(skymodel, res_images, noise_image):
    results = dict()
    beam = (20.0/3600, 20.0/3600, 0)
    noise_sig = noise_sigma(noise_image)
    model_lsm = Tigger.load(skymodel)
    model_sources = model_lsm.sources
    noise_hdu = fitsio.open(noise_image)
    noise_data = noise_hdu[0].data
    rad = lambda a: a*(180/np.pi) # convert radians to degrees
    for image in res_images:
        imager = image.split('_')[2].split('-')[0]
        results[imager] = []
        residual_hdu = fitsio.open(image)
        # Get the header data unit for the residual rms
        residual_data = residual_hdu[0].data
        res_header = residual_hdu[0].header
        nchan = (residual_data.shape[1] if residual_data.shape[0] == 1
                 else residual_data.shape[0])
        for model_source in model_sources:
            src  = model_source
            RA = rad(model_source.pos.ra)
            DEC = rad(model_source.pos.dec)
            # Cater for point sources and assume source extent equal to the
            # Gaussian major axis along both ra and dec axis
            dra = rad(src.shape.ex) if src.shape  else beam[0]
            ddec = rad(src.shape.ey) if src.shape  else beam[1]
            pa = rad(src.shape.pa) if src.shape  else beam[2]
            emin, emaj = sorted([dra, ddec])
            fits_info = fitsInfo(image)
            rgn = sky2px(fits_info["wcs"],RA,DEC,dra,ddec,fits_info["dra"], beam[1])
            imslice = slice(rgn[2], rgn[3]), slice(rgn[0], rgn[1])
            noise_area = noise_data[0,0,:,:][imslice]
            noise_rms = noise_area.std()
            # if image is cube then average along freq axis
            flux_std = 0.0
            flux_mean = 0.0
            for frq_ax in range(nchan):
                if residual_data.shape[0] == 1:
                    target_area = residual_data[0,frq_ax,:,:][imslice]
                else:
                    target_area = residual_data[frq_ax,0,:,:][imslice]
                flux_std += target_area.std()
                flux_mean += target_area.mean()
                if frq_ax == nchan - 1:
                    flux_std = flux_std/float(nchan)
                    flux_mean = flux_mean/float(nchan)
            results[imager].append([model_source.name, src.flux.I, flux_std,
                                    noise_rms, flux_std/noise_rms,
                                    src.flux.I/flux_std,
                                    src.flux.I/noise_sig, flux_mean,
                                    flux_mean/noise_rms])
    return results
