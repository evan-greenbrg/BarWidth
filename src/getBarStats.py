import argparse
import errno
import os
from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import gdal
import osr
import numpy as np
import pandas
from pyproj import Proj

from BarHandler import BarHandler


MIN_RSQUARE = 0.05
BAR_PARAM_FN = 'bar_parameters.csv'
BAR_DATA_FN = 'bar_data.csv'
RSQUARE_FN = 'rsquared_dataframe.csv'


def main():
    parser = argparse.ArgumentParser(description='template file resolver')
    parser.add_argument('input_param', type=argparse.FileType('r'))
    args = parser.parse_args()
    input_param = load(args.input_param, Loader=Loader)

    # Raise Errors if files don't exists
    if not os.path.exists(input_param['xPath']):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), input_param['xPath'])
    if not os.path.exists(input_param['coordPath']):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), input_param['coordPath'])
    if not os.path.exists(input_param['barPath']):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), input_param['barPath'])
    if not os.path.exists(input_param['demPath']):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), input_param['demPath'])
    if not input_param.get('depth'):
        raise NameError('No depth given')
    if not input_param.get('interpolate'):
        raise NameError('No interpolation direction given')
    if not input_param.get('mannual'):
        raise NameError('No mannual or automatic direction given given')
    if not input_param.get('outputRoot'):
        raise NameError('No output root given')

    # Load xsections and coordinates
    xsections = np.load(input_param['xPath'], allow_pickle=True)
    coordinates = pandas.read_csv(input_param['coordPath'])

    # Get Proj String
    ds = gdal.Open(input_param['demPath'], 0)
    ProjStr = "epsg:{0}".format(
        osr.SpatialReference(
            wkt=ds.GetProjection()
        ).GetAttrValue('AUTHORITY', 1)
    )
    myProj = Proj(ProjStr)

    # Initialize BarHandler
    bh = BarHandler()

    # Read in the bar file to find the channel bars
    print('Loading Bar .csv file')
    bar_df = pandas.read_csv(
        input_param['barPath'],
        names=[
            'Latitude_us',
            'Longitude_us',
            'Latitude_ds',
            'Longitude_ds'
        ],
        header=1
    )
    # Convert the Bar Lat Long to UTM Easting Northing
    print('Converting Bar Coordinates to Easting Northing')
    bar_df = bh.convert_bar_to_utm(myProj, bar_df)

    # unpersist gdal object
    ds = None

    # Make structure that contains the sections for each bar
    print('Making Bar section structure')
    bar_sections = {}
    for idx, bar in bar_df.iterrows():
        sections = bh.get_bar_xsections(
            coordinates,
            xsections,
            bar_df.iloc[idx]
        )
        print(len(sections))
        bar_sections[str(idx)] = sections

    # Initialize the parameters
    parameters = {
        'bar': [],
        'idx': [],
        'L': [],
        'X0': [],
        'k': [],
    }

    # Make structure that contains the sigmoids and only the bar side section
    print('Fitting sigmoid to channel bars')
    bar_widths = {}
    types = [
        ('location', 'object'),
        ('dem_width', 'f8'),
        ('water_width', 'f8'),
        ('sigmoid', 'object'),
        ('easting', 'object'),
        ('northing', 'object'),
        ('distance', 'object'),
        ('elevation', 'object'),
    ]

    # Make the dataframe that will keep track of the R-Squared
    rsquared_di = {
        'bar': [],
        'idx': [],
        'rsquared': [],
    }

    # Save for filtered/Saved stats
    filtered = 0
    saved = 0
    # Run through all of the bars and sections
    for bar, sections in bar_sections.items():
        widths = np.array([], dtype=types)
        for idx, section in np.ndenumerate(sections):
            if (
                section['dem_width'] == 'nan'
            ) or (
                not section['bank']
            ) or (
                section['water_width'] == 'nan'
            ):
                width = np.array(
                    tuple(
                        [
                            section[0],
                            section['dem_width'],
                            section['water_width'],
                            None,
                            section['elev_section']['easting'],
                            section['elev_section']['northing'],
                            section['elev_section']['distance'],
                            section['elev_section']['value_smooth']
                        ]
                    ),
                    dtype=widths.dtype
                )
                filtered += 1
            else:
                # interpolate profile down
                if input_param['interpolate']:
                    section = bh.interpolate_down(
                        input_param.get('depth'),
                        section
                    )

                # Find the minimum and shift the cross-section
                section = bh.shift_cross_section_down(
                    section,
                )

                if input_param['mannual']:
                    # Find the sigmoid equation
                    popt = bh.mannual_fit_bar(section)

                    # Find the r-squared
#                    rsquared = bh.get_r_squared(section, banks, popt)
                    rsquared = 1

                else:
                    # Find the side of the channel with the bar
                    banks = bh.find_bar_side(section['bank'])

                    # Flip cross-sections so they are facing the same way
                    section, banks = bh.flip_bars(section, banks)

                    # Find x for maximum slope and maximum slope value
                    x0, dydx = bh.find_maximum_slope(
                        section['elev_section'],
                        banks
                    )
                    # Fit sigmoid parameters
                    popt = bh.fit_sigmoid_parameters(
                        section,
                        banks,
                        x0,
                        dydx
                    )

                    # Get the R-Squared
                    rsquared = bh.get_r_squared(section, banks, popt)

                # Filter based on R-squared value
                if (rsquared < MIN_RSQUARE):
                    width = np.array(
                        tuple(
                            [
                                section[0],
                                section['dem_width'],
                                section['water_width'],
                                None,
                                section['elev_section']['easting'],
                                section['elev_section']['northing'],
                                section['elev_section']['distance'],
                                section['elev_section']['value_smooth']
                            ]
                        ),
                        dtype=widths.dtype
                    )
                    filtered += 1
                else:
                    # Keep track of the rsquared values
                    rsquared_di['bar'].append(bar)
                    rsquared_di['idx'].append('{0}_{1}'.format(bar, idx[0]))
                    rsquared_di['rsquared'].append(rsquared)

                    # Keep track of the parameters
                    parameters['bar'].append(bar)
                    parameters['idx'].append('{0}_{1}'.format(bar, idx[0]))
                    parameters['L'].append(popt[0])
                    parameters['X0'].append(popt[1])
                    parameters['k'].append(popt[2])

                    # store the sigmoid parameters and the cross section
                    width = np.array(
                        tuple(
                            [
                                section[0],
                                section['dem_width'],
                                section['water_width'],
                                popt,
                                section['elev_section']['easting'],
                                section['elev_section']['northing'],
                                section['elev_section']['distance'],
                                section['elev_section']['value_smooth']
                            ]
                        ),
                        dtype=widths.dtype
                    )
                    saved += 1

            widths = np.append(widths, width)

        bar_widths[bar] = widths

    # Save the parameter_df
    parameters_df = pandas.DataFrame(data=parameters)

    # Find the width and height of the channel bars
    print('Finding clinoform width and height')
    bar_data = {
        'bar': [],
        'idx': [],
        'easting': [],
        'northing': [],
        'channel_width_dem': [],
        'channel_width_water': [],
        'channel_width_mean': [],
        'bar_width': [],
        'bar_height': []
    }

    n = 0
    for bar, sections in bar_widths.items():
        L_mean = np.median(
            [i['sigmoid'][0] for i in sections if i['sigmoid']]
        )
        for idx, section in np.ndenumerate(sections):
            # Don't track if there is no channel width
            if str(section['dem_width']) == 'nan':
                continue

            # Filter out the ill-fit sigmoid parameters
            elif not section['sigmoid']:
                continue

            elif (section['sigmoid'][0] / L_mean) < 0.01:
                continue

            else:
                # Get the bar width from the sigmoid
                bar_width, bar_height = bh.get_bar_geometry(
                    section['distance'],
                    section['sigmoid']
                )

                try:
                    water_width = int(section['water_width'])
                except:
                    water_width = int(section['dem_width'])

                # Store data
                bar_data['bar'].append(bar)
                bar_data['idx'].append('{0}_{1}'.format(bar, idx[0]))
                bar_data['easting'].append(section['location'][0])
                bar_data['northing'].append(section['location'][1])
                bar_data['channel_width_dem'].append(int(section['dem_width']))
                bar_data['channel_width_water'].append(water_width)
                bar_data['channel_width_mean'].append(
                    (int(section['dem_width']) + water_width) / 2
                )
                bar_data['bar_width'].append(bar_width)
                bar_data['bar_height'].append(bar_height)
                n += 1

    # Create dataframes from data dicts
    parameters_df = pandas.DataFrame(parameters)
    bar_data_df = pandas.DataFrame(bar_data)
    rsquared_df = pandas.DataFrame(rsquared_di)

    # Save parameters data
    print('Saving bar parameters')
    parameters_df.to_csv(input_param['outputRoot'] + BAR_PARAM_FN)

    # Save the bar data
    print('Saving Bar Data')
    bar_data_df.to_csv(input_param['outputRoot'] + BAR_DATA_FN)

    # Save the r-squared data
    print('Saving R-Squared')
    rsquared_df.to_csv(input_param['outputRoot'] + RSQUARE_FN)

    print('Logging some stats:')
    print('Total Bars: {}'.format(len(bar_widths)))
    print('Total Cross-Sections: {}'.format(filtered + saved))
    print('Cross-Sections filtered out: {}'.format(filtered))
    print('Cross-Sections Saved: {}'.format(saved))


if __name__ == "__main__":
    main()
