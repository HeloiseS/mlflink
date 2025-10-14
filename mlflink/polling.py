""" Poll the Fink servers only once at a time """
import logging
from fink_client.consumer import AlertConsumer
from fink_client.visualisation import extract_field
import pandas as pd
import numpy as np
from datetime import datetime
import lasair
import os

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s [%(levelname)s] %(name)s.%(funcName)s: %(message)s")



def poll_n_alerts(myconfig, 
                  topics, 
                  n, 
                  outdir) -> None:
    """ Connect to and poll fink servers once.

    Parameters
    ----------
    myconfig: dic
        python dictionnary containing credentials
    topics: list of str
        List of string with topic names
    """

    logger.info(f"Polling {n} alerts from topics: {topics}")

    maxtimeout = 10
    prefix = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Instantiate a consumer
    try:
        consumer = AlertConsumer(topics, myconfig)
        out = consumer.consume(n, maxtimeout)
    except Exception as e:
        logger.error(f"Failed to connect or consume alerts: {e}")
        return

    if len(out) > 0:
        topics, alerts, keys = np.transpose(out) 
        logger.info(f"Received {len(alerts)} alerts.")
    else:
        logger.info(f"No alerts received in the last {maxtimeout} seconds.")
        consumer.close()
        return
    outpath = os.path.expanduser(outdir + f'{prefix}_alerts.parquet')
    alerts_df = pd.DataFrame.from_records(alerts)
    alerts_df.to_parquet(outpath)

    """
    try:
        alert_data = process_alerts(alerts)
        clean_dat = run_sherlock(alert_data)
        if clean_dat.empty:
            logger.info("No alerts left after filtering.")
        else:
            outpath = os.path.expanduser(outdir + f'{prefix}_alerts.parquet')
            clean_dat.to_parquet(outpath)
            logger.info(f"Saved {len(clean_dat)} cleaned alerts to {outpath}")
    except Exception as e:
        logger.error(f"Error during alert processing: {e}")
    """

    consumer.close()
    logger.info("Consumer connection closed.")
    return 

if __name__ == "__main__":
    """ Poll the servers only once at a time """
    # TODO make the outdir, the config, the topic and the n alerts parsable via CL or yaml

    outdir =  '~/Data/mlflink_test_stream/'
    # load user configuration
    # to fill
    myconfig = {
        'bootstrap.servers': 'kafka-ztf.fink-broker.org:24499',
        'group.id': 'heloise_finkvra'
    }

    topics = ['fink_vra_ztf']

    n_alerts = 50
    poll_n_alerts(myconfig, topics, n=n_alerts, outdir=outdir)
