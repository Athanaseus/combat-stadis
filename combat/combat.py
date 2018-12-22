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
    for model_source in model_sources:
        I_out = 0.0
        I_out_err = 0.0
        name = model_source.name
        RA = model_source.pos.ra
        DEC = model_source.pos.dec
        I_in = model_source.flux.I
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
    return targets_flux, targets_scale, targets_position


def get_ra_dec_range(area=1.0, phase_centre="J2000,0deg,-30deg"):
    """Get RA and DEC range from area of observations and phase centre"""
    ra = float(phase_centre.split(',')[1].split('deg')[0])
    dec = float(phase_centre.split(',')[2].split('deg')[0])
    d_ra = np.sqrt(area)/2
    d_dec = np.sqrt(area)/2
    ra_range = [ra-d_ra, ra+d_ra]
    dec_range = [dec-d_dec, dec+d_dec]
    return ra_range, dec_range
