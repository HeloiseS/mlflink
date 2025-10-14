"""Data processing module to make features from data saved by consumer"""
import pandas as pd
import logging
import numpy as np
from astropy.coordinates import SkyCoord
import os
from fink_client.visualisation import extract_field
import lasair

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s [%(levelname)s] %(name)s.%(funcName)s: %(message)s")

def raw2clean(alerts_df: pd.DataFrame) -> pd.DataFrame:
    """Process raw alerts into a cleaned DataFrame."""

    logger.info("Extracting columns of interest from alerts.")

    alerts_df['mag'] = alerts_df.apply(lambda alert: extract_field(alert, 'magpsf'), axis=1)
    alerts_df['maglim'] = alerts_df.apply(lambda alert: extract_field(alert, 'diffmaglim'), axis=1)
    alerts_df['mjd'] = alerts_df.apply(lambda alert: extract_field(alert, 'jd'), axis=1) - 2400000.5
    alerts_df['fid'] = alerts_df.apply(lambda alert: extract_field(alert, 'fid'), axis=1)
    alerts_df['isdiffpos'] = alerts_df.apply(lambda alert: extract_field(alert, 'isdiffpos'), axis=1)
    alerts_df['ra'] = alerts_df.apply(lambda row: row['candidate']['ra'], axis=1)
    alerts_df['dec'] = alerts_df.apply(lambda row: row['candidate']['dec'], axis=1)
    alerts_df['drb'] = alerts_df.apply(lambda row: row['candidate']['drb'], axis=1)


    logger.info("Alerts processed into dataframe: %d", len(alerts_df))

    return alerts_df[[
        'candid', 'objectId', 'ra', 'dec', 'drb',
        'mjd', 'mag', 'maglim', 'fid', 'isdiffpos',
        'lc_features_g', 'lc_features_r'
    ]]


def run_sherlock(alert_data:pd.DataFrame):
    """Run Sherlock on the alert data processed by process_data."""
    
    if "LASAIR_TOKEN" not in os.environ:
        alert_data['sherl_class'] = np.nan
        alert_data['sep_arcsec'] = np.nan
        return alert_data
    
    logger.info("Running Sherlock classification on alerts.")
    # the lasair client will be used for fetching Sherlock data
    L = lasair.lasair_client(os.environ.get('LASAIR_TOKEN'), 
                             endpoint='https://lasair-ztf.lsst.ac.uk/api')
    
    sherl_class = []
    sherl_separcsec = []

    for i in range(alert_data.shape[0]):
        _sherl = L.sherlock_position(alert_data.iloc[i]['ra'], alert_data.iloc[i]['dec'], lite=False)
        sherl_class.append(_sherl['classifications']['transient_00000'][0])
        try:
            sherl_separcsec.append(_sherl['crossmatches'][0]['separationArcsec'])
        except IndexError:
            # If orphan will get no match 
            sherl_separcsec.append(np.nan)

    logger.info("Successfully retrieved Sherlock classifications for %d alerts.", len(sherl_class))
    data_w_sherl = alert_data.join(pd.DataFrame(np.atleast_2d([sherl_class, sherl_separcsec]).T, 
                                                columns=['sherl_class', 'sep_arcsec']))
    
    # remove AGNs and Variable Stars
    mask = (data_w_sherl['sherl_class'] == 'AGN') | (data_w_sherl['sherl_class'] == 'VS')
    clean_data = data_w_sherl[~mask]
    logger.info("After removing AGNs and Variable Stars, %d alerts remain.", clean_data.shape[0])
    return clean_data
