import os
import csv
import glob
import json
import math
import logging
import warnings
import argparse
import inflection
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from PyPDF2 import PdfReader, PdfWriter, PageObject, Transformation

warnings.filterwarnings("ignore", category=Warning)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def scale_curr (curr, meastemp, scaletemp):

    T1 = meastemp + 273.0
    T2 = scaletemp + 273.0
    Eg = 1.21 #eV
    k = 1.381e-23 #V/C
    q = 1.601e-19 #C
    f = 1/(((T1/T2)**2)*np.exp(-1*(Eg*q/k)*(1/T1 - 1/T2)))
    #print ("f2", f)
    #print ("Eg term", Eg*q/k, "T term", T1/T2, 1/T1 - 1/T2)
    #print ("Old/New Current", curr, curr * f)
    return (abs(curr*f))

def calc_mad (df):

    med_df = df.median()
    meds = []
    for i in df:
        meds.append(abs(med_df - i))
    return (np.median(meds))

def compute_profile(x, y, nbin=(100,100)):

    # use of the 2d hist by numpy to avoid plotting
    h, xe, ye = np.histogram2d(x,y,nbin)

    # bin width
    xbinw = xe[1]-xe[0]

    # getting the mean and RMS values of each vertical slice of the 2D distribution
    # also the x valuse should be recomputed because of the possibility of empty slices
    x_array      = []
    x_slice_mean = []
    x_slice_rms  = []
    for i in range(xe.size-1):
        yvals = y[ (x>xe[i]) & (x<=xe[i+1]) ]
        if yvals.size>0: # do not fill the quanties for empty slices
            x_array.append(xe[i]+ xbinw/2)
            x_slice_mean.append( yvals.mean())
            x_slice_rms.append( yvals.std())
    x_array = np.array(x_array)
    x_slice_mean = np.array(x_slice_mean)
    x_slice_rms = np.array(x_slice_rms)

    return x_array, x_slice_mean, x_slice_rms


def get_vdep(df: pd.DataFrame, guess: int) -> tuple[float, list]:
    """Perform a curve fit to the provided curve and output the depletion voltage as well as
    optional fit parameters.

    Args:
        df (pd.DataFrame): The provided curve to fit the data to
        guess (int): Initial guess of the depletion voltage

    Returns:
        tuple[float, list]: The depletion voltage and the fit parameters.
        """

    voltage = abs(np.array(df["VOLTS"]))
    capacitance = np.array(1 / (df["CAPCTNC_PFRD"] / 1e12) ** 2)
    popt, pcov = curve_fit(
        piecewise_linear,
        np.abs(voltage[1:]),
        capacitance[1:],
        p0=[guess, 1e7, capacitance[-3], -1e14],
    )
    return np.absolute(popt[0]), popt


def piecewise_linear(x: np.ndarray, x0: float, b: float, k1: float, k2: float) -> np.ndarray:
    """ Define piecewise linear function for the extraction of the depletion voltage. """

    condlist = [x < x0, (x >= x0)]
    funclist = [lambda x: k1 * x + b, lambda x: k1 * x0 + b + k2 * (x - x0)]
    return np.piecewise(x, condlist, funclist)


def save_dframe(df: pd.DataFrame, sensor: str, measurement_type: str, date: str, run_type: str, struct):
    """Save dataframe to CSV file.

    Args:
        df (pd.DataFrame): Dataframe to be saved
        sensor (str): Name of the tested sensor
        measurement_type (str): Measurement type
        date (str): Date on which the measurement was performed
        run_type (str): Type of run (SQC or VQC)
    """

    path = f"./data/{sensor[0:5]}/{sensor}/{measurement_type}"
    if not os.path.exists(path):
        os.makedirs(path)
    date = date.replace(" ", "-").replace(":", "-")
    df.to_csv(f"{path}/{run_type}_{sensor}_{struct}_{measurement_type}_{date}.csv", index=False)


def organize_dframe(data: list, sensor: str, measurement_type: str, run_type: str, struct):
    """ Takes the data received from the query, formats and saves it.

    Args:
        data (list): List of strings received from the query
        sensor (str): Name of the sensor the data belongs to
        measurement_type (str): Type of the measurement
        run_type (str): Type of run (SQC or VQC)

    """

    structname = struct[0]
    if struct[1] == "T":  structname += "_GR"

    # Create a df from the query result
    data = [item.split(",") for item in data]
    header = data[0]
    data = pd.DataFrame(data[1:-1], columns=header)
    for column in ["VOLTS", "STRIP", "STRIPCOUPLE"]:
        try:
            data = data.dropna(subset=[column])
        except KeyError:
            continue

    # Convert the df columns to the correct data type
    if measurement_type != 'alibava':
      dtypes = [
        "int32",
        "float64",
        "float64",
        "object",
        "float64",
        "float64",
        "float64",
        "int32",
        "object",
        "object",
        "object",
        "int32",
        "int32",
      ]
      if run_type == 'IT':
        dtypes.extend(["object", "object", "float64", "float64", "float64"])
    else:
      dtypes = [
        "int32",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "int32",
        "object",
        "object",
        "object",
        "int32",
        "int32",
      ]
      dtypes.extend(["object", "object", "float64", "float64", "float64"])

    for i, key in enumerate(header):
        try:
            data[key] = data[key].astype(dtypes[i])
        except ValueError:
            continue

    # Check for different measurement dates and sort them if needed
    mdates = data["BEGIN_DATE"].unique()
    mdates_sorted = sorted(mdates, reverse=True)
    logging.info(f"found {len(mdates)} {run_type} {measurement_type} measurement(s) for sensor {sensor}")

    # Iterate through the measurement dates and save the corresponding df
    if measurement_type == 'alibava':
        mvolts = data["VOLTAGE_V"].unique()
        logging.info(f"found {len(mvolts)} voltages {run_type} {measurement_type} measurement(s) for sensor {sensor}")
        for voltage in mvolts:
            temp = data[data["VOLTAGE_V"] == voltage]
            temp.sort_values(by="ANN_TIME_H_21C", inplace=True)
            save_dframe(temp, sensor, measurement_type, str(voltage).replace('.0',''), run_type, structname)
    else:
      for mdate in mdates_sorted:
        temp = data[data["BEGIN_DATE"] == mdate]
        temp.reset_index(drop=True, inplace=True)

        if measurement_type in ["iv", "cv"]:
            temp = temp.sort_values(by="VOLTS", ascending=False)
            save_dframe(temp, sensor, measurement_type, mdate, run_type, structname)

        elif measurement_type in ["r_int", "c_int"]:
            temp.sort_values(by="TIME", ascending=False, inplace=True)
            temp.drop_duplicates(subset="STRIPCOUPLE", keep="first", inplace=True)
            temp = temp.sort_values(by="STRIPCOUPLE")
            temp = temp[~(temp["STRIPCOUPLE"].isin([960, 1920, 1016, 2032]))]
            save_dframe(temp, sensor, measurement_type, mdate, run_type, structname)
        #elif measurement_type == 'alibava':
        else:
            temp.sort_values(by="TIME", ascending=False, inplace=True)
            temp.drop_duplicates(subset="STRIP", keep="first", inplace=True)
            temp = temp.sort_values(by="STRIP")
            save_dframe(temp, sensor, measurement_type, mdate, run_type, structname)


def get_meas_files(sensor: str, model, runtypes, structs):
    """Query all SQC measurement data available in the CMS DB

    Args:
        sensor (str): Name of the sensor for which the measurement data is queried
    """

    # Reading the config file containing information about all measurement types
    #with open("./models/KindOfMeasurement_IT.json", "r") as file:
    with open("./models/"+model, "r") as file:
        measurements = json.load(file)

    # query the data for each SQC measurement type
    #print (runtypes, structs)
    for run_type in runtypes:
      for struct in structs:
        for meas in measurements:
            #print ("Meas", meas['mtype'], run_type)
            if meas['mtype'] not in ['iv', 'cv'] and run_type == 'VQC':
                continue
            if meas['mtype'] == 'scaled_iv':
                continue
            elif run_type == "IT" and meas['mtype'] != 'alibava' and struct[0] == 'BABYSENSOR':
                #logging.info(f"searching for {run_type} {meas['mtype']} measurements for IT sensors {sensor}")
                query = f"select c.PART_ID, c.{inflection.underscore(meas['fields']['cmsX'])},  \
                    c.{inflection.underscore(meas['fields']['cmsY'])}, \
                    c.TIME, c.TEMP_DEGC, c.RH_PRCNT, c.BIASCURRNT_NAMPR, \
                    c.CONDITION_DATA_SET_ID, d.LOCATION, d.RUN_TYPE, \
                    d.BEGIN_DATE, d.RUN_NUMBER, p.BATCH_NUMBER, e.KIND_OF_HM_STRUCT_ID, e.RADIATION_TYP, e.TARGET_FLUENCE, e.MEASURED_FLUENCE, e.ANN_TIME_H_21C from trker_cmsr.c{meas['fields']['ID']} \
                    c inner join trker_cmsr.parts p on p.ID = c.PART_ID inner join trker_cmsr.datasets \
                    d on d.ID = c.CONDITION_DATA_SET_ID inner join trker_cmsr.c15400 e on e.CONDITION_DATA_SET_ID = c.AGGREGATED_COND_DATA_SET_ID \
                    where p.NAME_LABEL='{sensor}' and e.KIND_OF_HM_STRUCT_ID='{struct[0]}'\
                    and d.RUN_TYPE='{run_type}' order by p.ID, c.time ASC"
            elif run_type == "IT" and meas['mtype'] != 'alibava':
                #logging.info(f"searching for {run_type} {meas['mtype']} measurements for IT sensors {sensor}")
                query = f"select c.PART_ID, c.{inflection.underscore(meas['fields']['cmsX'])},  \
                    c.{inflection.underscore(meas['fields']['cmsY'])}, \
                    c.TIME, c.TEMP_DEGC, c.RH_PRCNT, c.BIASCURRNT_NAMPR, \
                    c.CONDITION_DATA_SET_ID, d.LOCATION, d.RUN_TYPE, \
                    d.BEGIN_DATE, d.RUN_NUMBER, p.BATCH_NUMBER, e.KIND_OF_HM_STRUCT_ID, e.RADIATION_TYP, e.TARGET_FLUENCE, e.MEASURED_FLUENCE, e.ANN_TIME_H_21C from trker_cmsr.c{meas['fields']['ID']} \
                    c inner join trker_cmsr.parts p on p.ID = c.PART_ID inner join trker_cmsr.datasets \
                    d on d.ID = c.CONDITION_DATA_SET_ID inner join trker_cmsr.c15400 e on e.CONDITION_DATA_SET_ID = c.AGGREGATED_COND_DATA_SET_ID \
                    where p.NAME_LABEL='{sensor}' and e.KIND_OF_HM_STRUCT_ID='{struct[0]}' and e.GR_CONNECTED='{struct[1]}'\
                    and d.RUN_TYPE='{run_type}' order by p.ID, c.time ASC"
            elif run_type == "IT" and meas['mtype'] == 'alibava':
                #logging.info(f"searching for {run_type} {meas['mtype']} measurements for Alibava sensors {sensor}")
                query = f"select c.PART_ID, c.VOLTAGE_V, c.SEEDQ_MPV_E, c.SEEDQ_MPV_ADC, c.SEEDQ_MED_E, c.SEEDQ_MED_ADC,  \
                    c.CLUSTQ_MPV_E, c.CLUSTQ_MPV_ADC, c.CLUSTQ_MED_E, c.CLUSTQ_MED_ADC, c.LEAKAGE_CURR_A, \
                    c.NOISE_AVG, c.GAIN_AVG, c.CMNOISE_AVG, c.CMNOISE_STD, \
                    c.CONDITION_DATA_SET_ID, d.LOCATION, d.RUN_TYPE, \
                    d.BEGIN_DATE, d.RUN_NUMBER, p.BATCH_NUMBER, e.KIND_OF_HM_STRUCT_ID, e.RADIATION_TYP, e.TARGET_FLUENCE, e.MEASURED_FLUENCE, e.ANN_TIME_H_21C from trker_cmsr.c{meas['fields']['ID']} \
                    c inner join trker_cmsr.parts p on p.ID = c.PART_ID inner join trker_cmsr.datasets \
                    d on d.ID = c.CONDITION_DATA_SET_ID inner join trker_cmsr.c15400 e on e.CONDITION_DATA_SET_ID = c.AGGREGATED_COND_DATA_SET_ID \
                    where p.NAME_LABEL='{sensor}' \
                    and d.RUN_TYPE='{run_type}' order by p.ID"
            else:
                #logging.info(f"searching for {run_type} {meas['mtype']} measurements for sensors {sensor}")
                query = f"select c.PART_ID, c.{inflection.underscore(meas['fields']['cmsX'])},  \
                    c.{inflection.underscore(meas['fields']['cmsY'])}, \
                    c.TIME, c.TEMP_DEGC, c.RH_PRCNT, c.BIASCURRNT_NAMPR, \
                    c.CONDITION_DATA_SET_ID, d.LOCATION, d.RUN_TYPE, \
                    d.BEGIN_DATE, d.RUN_NUMBER, p.BATCH_NUMBER from trker_cmsr.c{meas['fields']['ID']} \
                    c inner join trker_cmsr.parts p on p.ID = c.PART_ID inner join trker_cmsr.datasets \
                    d on d.ID = c.CONDITION_DATA_SET_ID \
                    where p.NAME_LABEL='{sensor}' \
                    and d.RUN_TYPE='{run_type}' order by p.ID, c.time ASC"

            #print ("SQL:", "{}".format(query))
            initial_path = os.getcwd()
            os.chdir('./resthub/clients/python/src/main/python/')
            p1 = subprocess.run(
                [
                    "python3",
                    "rhapi.py",
                    "-n",
                    "--krb",
                    "-all",
                    "--url=http://dbloader-tracker:8113",
                    "{}".format(query),
                ],
                capture_output=True,
            )
            data = p1.stdout.decode().splitlines()
            os.chdir(initial_path)
            organize_dframe(data, sensor, meas["mtype"], run_type, struct)
            if run_type == "IT" and meas['mtype'] == 'alibava':
                continue


def stich_summary_pdf(folder, outfileprefix, filterstring):
    """ Creates a summary pdf page fromm all plots created for one sensor

    Args:
        sensor (str): Name of the sensor
    """

    # Search for the individual plots with extension .pdf and sort them by the substring list
    #batch = sensor[:5]
    substring_list = ["pin", "r_int", "c_int", "i_leak", "r_poly", "cc", "iv", "cv"]
    search_pattern = os.path.join(f"./plots/{folder}", filterstring)
    pdf_filenames = glob.glob(search_pattern)
    print ("Saving Plots:", pdf_filenames)
    pdf_filenames = list(filter(lambda item: "summary" not in item, pdf_filenames))
    pdf_filenames = sorted(
        pdf_filenames,
        key=lambda x: [
            substring_list.index(substring)
            for substring in substring_list
            if substring in x
        ],
    )

    # Create a list to store the PdfReader objects for each file
    inputs = []

    # Iterate over the pdf_filenames list and create PdfFileReader objects
    for filename in pdf_filenames:
        input_pdf = PdfReader(open(filename, "rb"), strict=False)
        inputs.append(input_pdf)

    # Initialize total width and height
    total_width = 0
    total_height = 0

    # Calculate total width and maximum height from the input pages
    for input_pdf in inputs:
        page = input_pdf.pages[0]
        if len(inputs) == 1:
            total_width = page.mediabox.upper_right[0] * 1
        elif len(inputs) == 2:
            total_width = page.mediabox.upper_right[0] * 2
        else:
            total_width = page.mediabox.upper_right[0] * 3
        total_height = math.ceil(len(pdf_filenames) / 3) * (
            page.mediabox[3] - page.mediabox[1]
        )

    # Create a blank page with the calculated dimensions
    new_page = PageObject.create_blank_page(None, total_width, total_height)

    # Merge the input pages onto the new page
    x_offset = 0
    y_offset = 0
    for i, input_pdf in list(enumerate(inputs)):
        page = input_pdf.pages[0]
        if i != 0:
            x_offset += page.mediabox.upper_right[0]
        if i % 3 == 0 and i != 0:
            x_offset = 0
            y_offset += page.mediabox[3] - page.mediabox[1]

        page.add_transformation(Transformation().translate(x_offset, y_offset), expand=True)
        new_page.merge_page(page)

    # Save the output to a new PDF file
    if len(inputs) > 0:
        output_filename = f"./plots/{folder}/" + outfileprefix + "summary.pdf"
        output = PdfWriter()
        output.add_page(new_page)
        output.write(open(output_filename, "wb"))


def create_plots(sensorlist, model, runtypes, folder, label, status, ali_voltages, structs):
  """Create and save plots of all SQC parameters based on measurements found in the CMS DB
    Args:
      sensor (str): Name of the sensor
  """

  print ("Making Plots")
  # Create diretory for the plots if needed
  #batch = sensor[0:5]
  #path = f"./plots/{batch}/{sensor}"
  profdata = []
  #df_append = pd.DataFrame()
  path = f"./plots/{folder}"
  if not os.path.exists(path):
      os.makedirs(path)

  # Read the config containing information for the plots
  with open("./models/" + model, "r") as file:
      measurements = json.load(file)

  colors = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
  #plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b'], marker=['o', '+', 'x'])))
  labellist = []
  for meas in measurements:

    # Look through data to determine if plot should be log/linear
    log_mode = False
    outlier = False
    min_data = 1e38
    max_data = 0
    #print (meas, sensorlist, structs, ali_voltages)
    for sensor in sensorlist:
     for struct in structs:
      structname = struct[0]
      if struct[1] == "T": structname += "_GR"
      for voltage in ali_voltages:
        #print ("v", voltage)
        batch = sensor[0:5]
        measurement_type = meas["mtype"]
        if measurement_type == "scaled_iv":
            meastype = "iv"
        else:
            meastype = measurement_type
        try:
            files = os.listdir(f"./data/{batch}/{sensor}/{meastype}")
            dates = [file[-23:-4] for file in files]
            run_types = [file[0:3] for file in files]
            #print ("DatesRun", dates, run_types)
        except FileNotFoundError:
            continue
        for i, date in enumerate(dates):
            run_type = run_types[i]
            if "DIODE_HALF_PSTOP_GR" in files[i]:
                struct_type = "DIODE_HALF_PSTOP_GR"
            elif "DIODE_HALF_PSTOP" in files[i]:
                struct_type = "DIODE_HALF_PSTOP"
            elif "DIODE_HALF" in files[i]:
                struct_type = "DIODE_HALF"
            elif "DIODE_QUARTER" in files[i]:
                struct_type = "DIODE_QUARTER"
            elif "DIODE" in files[i]:
                struct_type = "DIODE"
            elif "BABYSENSOR" in files[i]:
                struct_type = "BABYSENSOR"
            else:
                struct_type = "SENSOR"

            #print ("YoYo", struct_type, struct, date, structs, status, run_type, runtypes, measurement_type)
            if run_type == "IT_": run_type = "IT"
            if run_type == "mod": run_type = "mod_tailEncapsulated"
            if not(run_type in  runtypes):
                continue
            if structname != struct_type:
                continue
            if run_type == 'VQC' and measurement_type not in ['iv', 'cv']:
                continue
            if measurement_type == 'alibava':
                df = pd.read_csv(
                    f"./data/{batch}/{sensor}/{meastype}/{run_type}_{sensor}_{structname}_{meastype}_{voltage}.csv"
                )
            else:
                #print (batch, sensor, meastype, run_type, structname, meastype, date)
                df = pd.read_csv(
                    f"./data/{batch}/{sensor}/{meastype}/{run_type}_{sensor}_{structname}_{meastype}_{date}.csv"
                )
            #print (struct_type, struct, date, structs, status, df["MEASURED_FLUENCE"][0])
            #df_append = pd.concat([df_append, df], ignore_index=True)
            if run_type == "IT":
                if not('p' in status) and float(df["MEASURED_FLUENCE"][0]) == 0:
                    continue
                if not('i' in status) and float(df["ANN_TIME_H_21C"][0]) < 300 and float(df["ANN_TIME_H_21C"][0]) > 0:
                    continue
                if not('a' in status) and float(df["ANN_TIME_H_21C"][0]) > 300:
                    continue
            med_data = abs(df[meas["fields"]["plotcolumnY"]].median())
            mn_data = abs(df[meas["fields"]["plotcolumnY"]].mean())
            std_data = abs(df[meas["fields"]["plotcolumnY"]].std())
            min_data = min(min_data, med_data)
            max_data = max(max_data, med_data)

    #After reading all the data set plot limits
    max_min_ratio = np.divide(max_data, min_data, where=min_data != 0)
    if max_min_ratio > 200:
        log_mode = True
    #else:
        #print ("yo", sensor, voltage, abs(df[meas["fields"]["plotcolumnY"]]))
        #ypbot = np.percentile(abs(df[meas["fields"]["plotcolumnY"]]), 1)
        #yptop = np.percentile(abs(df[meas["fields"]["plotcolumnY"]]), 99)
        #ypad = 0.5*(yptop - ypbot)
        #ymin = max(0, ypbot - ypad)
        #ymax = yptop + ypad
    #df_append.to_csv(f"{path}/xy_{measurement_type}_{meas['fields']['name']}.csv")

    # Iterate through data files and check which measurement types are available.
    # For each available measurement type a plot is created and saved as pdf.
    for sensor in sensorlist:
     for struct in structs:
      structname = struct[0]
      if struct[1] == "T": structname += "_GR"
      for voltage in ali_voltages:
        batch = sensor[0:5]
        sensor_num = sensor[6:9]
        sensor_type = sensor[10:13].replace('-','')
        measurement_type = meas["mtype"]
        if measurement_type == "scaled_iv":
            meastype = "iv"
        else:
            meastype = measurement_type
        #print (batch, sensor, measurement_type, meastype)
        try:
            files = os.listdir(f"./data/{batch}/{sensor}/{meastype}")
            dates = [file[-23:-4] for file in files]
            run_types = [file[0:3] for file in files]
            #print ("Dates", dates)
        except FileNotFoundError:
            continue

        for i, date in enumerate(dates):
            run_type = run_types[i]
            if "DIODE_HALF_PSTOP_GR" in files[i]:
                struct_type = "DIODE_HALF_PSTOP_GR"
            elif "DIODE_HALF_PSTOP" in files[i]:
                struct_type = "DIODE_HALF_PSTOP"
            elif "DIODE_HALF" in files[i]:
                struct_type = "DIODE_HALF"
            elif "DIODE_QUARTER" in files[i]:
                struct_type = "DIODE_QUARTER"
            elif "DIODE" in files[i]:
                struct_type = "DIODE"
            elif "BABYSENSOR" in files[i]:
                struct_type = "BABYSENSOR"
            else:
                struct_type = "SENSOR"

            #print (structname, struct_type, struct, date, structs, status, run_type, runtypes, measurement_type)
            if run_type == "IT_": run_type = "IT"
            if run_type == "mod": run_type = "mod_tailEncapsulated"
            if not(run_type in  runtypes):
                continue
            if run_type == 'VQC' and measurement_type not in ['iv', 'cv']:
                continue
            if structname != struct_type:
                continue
            #print (f"./data/{batch}/{sensor}/{meastype}/{run_type}_{sensor}_{structname}_{meastype}_{date}.csv")
            if measurement_type == 'alibava':
                df = pd.read_csv(
                    f"./data/{batch}/{sensor}/{meastype}/{run_type}_{sensor}_{structname}_{meastype}_{voltage}.csv"
                )
            else:
                df = pd.read_csv(
                    f"./data/{batch}/{sensor}/{meastype}/{run_type}_{sensor}_{structname}_{meastype}_{date}.csv"
                )
            #print ("Read in pandas df")
            if run_type == "IT":
                if not('p' in status) and float(df["MEASURED_FLUENCE"][0]) == 0:
                    continue
                if not('i' in status) and float(df["ANN_TIME_H_21C"][0]) < 300 and float(df["ANN_TIME_H_21C"][0]) > 0:
                    continue
                if not('a' in status) and float(df["ANN_TIME_H_21C"][0]) > 300:
                    continue
                #print ("Ann:", df["ANN_TIME_H_21C"][0])
                if float(df["MEASURED_FLUENCE"][0]) == 0:
                    anntxt = "Preirrad"  
                elif float(df["ANN_TIME_H_21C"][0]) == 71.19:
                    anntxt = "Postirrad"  
                elif float(df["ANN_TIME_H_21C"][0]) == 284.76:
                    anntxt = "Postirrad"  
                elif float(df["ANN_TIME_H_21C"][0]) > 3500:
                    anntxt = "Postann"
                else:
                    anntxt = "Other"  
            labeltxt = ""
            for ltr in label:
                if ltr == 'b':  labeltxt += batch + "_"
                if ltr == 's':  labeltxt += sensor + "_"
                if ltr == 'k':  labeltxt += structname + "_"
                if ltr == 'm':  labeltxt += measurement_type + "_"
                if ltr == 'w':  labeltxt += sensor_num + "_"
                if ltr == 'd':  labeltxt += date[:-3] + "_"
                if ltr == 'D':  labeltxt += date[:-9] + "_"
                if ltr == 't':  labeltxt += sensor_type + "_"
                if ltr == 'n':  labeltxt += df["RADIATION_TYP"][0] + "_"
                if ltr == 'v':  labeltxt += voltage + "_"
                if ltr == 'a':  labeltxt += anntxt + "_"
                if ltr == 'r':  labeltxt += run_type + "_"
            labeltxt = labeltxt[:-1]
            #print ("Label", labeltxt)
            if not labeltxt in labellist:
                labellist.append(labeltxt)

            #min_data = min(min_data, min(abs(df[meas["fields"]["plotcolumnY"]])))
            #max_data = max(max_data, max(abs(df[meas["fields"]["plotcolumnY"]])))
            #ymean = df[meas["fields"]["plotcolumnY"]].mean()
            #yrms = df[meas["fields"]["plotcolumnY"]].std()
            #print ("Start plots", meas, sensor, run_type, min_data, max_data)
            plt.figure(1)
            #print ("meastype", measurement_type)
            if measurement_type == "scaled_iv":
                meastemp = df["TEMP_DEGC"][0]
                #print ("Scaled IV", meastemp)
                plt.plot(abs(df[meas["fields"]["plotcolumnX"]].values), scale_curr(df[meas["fields"]["plotcolumnY"]].values, meastemp, 20), marker="o", label=labeltxt, color=colors[labellist.index(labeltxt)%len(colors)])
                plt.xlabel(meas["fields"]["plotlabelX"], fontsize=16)
                plt.ylabel(meas["fields"]["plotlabelY"], fontsize=16)
                plt.title((meas["fields"]["plottitle"]), fontsize=16)
            elif measurement_type == "cv":
                plt.plot(
                    abs(df[meas["fields"]["plotcolumnX"]].values),
                    1
                    / (
                        (
                            df[meas["fields"]["plotcolumnY"]].values
                            * meas["fields"]["scalingfactor"]
                        )
                        ** 2
                    ),
                    marker="o",
                    label=labeltxt,
                    color=colors[labellist.index(labeltxt)%len(colors)]
                )
                if (run_type != 'VQC') and (len(sensorlist) == 1) and (len(dates) == 1):
                    vdep, popt = get_vdep(df, 300)
                    voltages = np.linspace(0, 600, 100)
                    cv_fit_values = piecewise_linear(voltages, *popt)
                    plt.plot(voltages, cv_fit_values, color="red")
                    plt.vlines(
                        x=vdep, ymin=0, ymax=1.1 * max(cv_fit_values), linestyle="dashed", color="red"
                    )
                    plt.ylim(0, 1.1 * max(cv_fit_values))
                    plt.text(
                        370, 0.7 * max(cv_fit_values), "Depletion Voltage:", fontsize=14, color="red"
                    )
                    plt.text(
                        370,
                        0.63 * max(cv_fit_values),
                        f"{np.round(vdep,1)}V",
                        fontsize=14,
                        color="red",
                    )
                plt.xlabel(meas["fields"]["plotlabelX"], fontsize=16)
                plt.ylabel(meas["fields"]["plotlabelY"], fontsize=16)
                plt.title((meas["fields"]["plottitle"]), fontsize=16)
            else:
                #max_min_ratio = np.divide(max_data, min_data, where=min_data != 0)
                #if max_min_ratio > 200:
                #    log_mode = True
                if (log_mode and max_min_ratio != np.inf):
                    plt.semilogy(
                        abs(df[meas["fields"]["plotcolumnX"]]).values,
                        abs(df[meas["fields"]["plotcolumnY"]].values)
                        * meas["fields"]["scalingfactor_log"],
                        marker="o",
                        label=labeltxt,
                        color=colors[labellist.index(labeltxt)%len(colors)]
                    )
                    plt.xlabel(meas["fields"]["plotlabelX"], fontsize=16)
                    plt.title(meas["fields"]["plottitle"], fontsize=16)
                    if meas["fields"]["scalingfactor_log"] == meas["fields"]["scalingfactor"]:
                        plt.ylabel(meas["fields"]["plotlabelY"], fontsize=16)
                    else:
                        plt.ylabel(meas["fields"]["plotlabelY_log"], fontsize=16)
                else:
                    plt.plot(
                        abs(df[meas["fields"]["plotcolumnX"]].values),
                        abs(df[meas["fields"]["plotcolumnY"]].values)
                        * meas["fields"]["scalingfactor"],
                        marker="o",
                        label=labeltxt,
                        color=colors[labellist.index(labeltxt)%len(colors)]
                    )
                    #plt.ylim(ymin,ymax)
                    plt.ylabel(meas["fields"]["plotlabelY"], fontsize=16)
                    plt.xlabel(meas["fields"]["plotlabelX"], fontsize=16)
                    plt.title(meas["fields"]["plottitle"], fontsize=16)
                if "profileplotx" in meas["fields"]:
                    #print ("Do profile plot ", meas["fields"]["profileplot"])
                    xp = df[meas["fields"]["profileplotx"]][0]
                    ymed = df[meas["fields"]["plotcolumnY"]].median()
                    ymad = calc_mad(df[meas["fields"]["plotcolumnY"]])
                    ymean = df[meas["fields"]["plotcolumnY"]].mean()
                    yrms = df[meas["fields"]["plotcolumnY"]].std()
                    plt.figure(2)
                    if "profileploty" in meas["fields"]:
                        if meas["fields"]["profileploty"] == 'median':
                            plt.errorbar(xp, ymed, ymad,fmt='_', ecolor='r', color='r')
                        elif meas["fields"]["profileploty"] == 'mean':
                            plt.errorbar(xp, ymed, ymad,fmt='_', ecolor='r', color='r')
                    #prof = fig.add_subplot(1, 1, 1)
                    #label=f'{sensor_num}, {date[:-3]} ({run_type})'
                    if run_type == 'IT':
                        profdata.append([sensor, batch, sensor[10:13], struct[0], df["LOCATION"][0], date, df["RADIATION_TYP"][0], voltage, anntxt, df["ANN_TIME_H_21C"][0], xp, ymean, ymed, yrms, ymad])
                    else:
                        profdata.append([sensor, batch, sensor[10:13], struct[0], df["LOCATION"][0], date, xp, ymean, ymed, yrms, ymad])

    plt.figure(1)
    plt.tight_layout()
    plt.grid()
    #plt.legend()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    #for ax in plt.gca().axes:
    #    ax.get_lines()[0].set_color("black")
    #for ln in plt.gca().get_lines():
    #    ln.set_color()
    if measurement_type == 'alibava':
        plt.savefig(f"{path}/xy_{measurement_type}_{meas['fields']['name']}.pdf")
    else:
        plt.savefig(f"{path}/xy_{measurement_type}.pdf")
    plt.clf()

    if "profileplotx" in meas["fields"]:
        plt.figure(2)
        plt.ylabel(meas["fields"]["plotlabelY"], fontsize=16)
        plt.xlabel(meas["fields"]["profileplotx"], fontsize=16)
        plt.title(meas["fields"]["plottitle"], fontsize=16)
        #prof.set_ylabel(meas["fields"]["plotlabelY"], fontsize=16)
        #prof.set_xlabel(meas["fields"]["profileplot"], fontsize=16)
        #prof.set_title(meas["fields"]["plottitle"], fontsize=16)
        plt.tight_layout()
        plt.grid()
        #plt.legend() 
        plt.savefig(f"{path}/pro_{measurement_type}.pdf")
        plt.clf()

        #Save csv file
        fields = ["Sensor", "Batch", "Type", "Structure", "Location", "Date", "Particle", "Voltage", "Annealing", "Ann (Hr 21C)", meas["fields"]["profileplotx"], "Mean", "Median", "StDev", "MAD"]
        with open(f"{path}/pro_{measurement_type}.csv", 'w') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)
            csvwriter.writerows(profdata)

    # Stich the individual pdf to a sensor summary pdf page
  stich_summary_pdf(folder, "", "xy_*.pdf")
  stich_summary_pdf(folder, "profile", "pro_*.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", help="batch name", action='append', nargs='+', required=False)
    parser.add_argument("-s", help="sensor name", required=False)
    parser.add_argument("-g", help="Plot labels, (b)atch No, (w)afer No, (D)ate, (d)ate + time, (s)ensor name, (t)ype[2S/PSS], (k)ind of structure, (n)/p, (a)nn", default = "bw")
    parser.add_argument("-m", help="(c)VIV, (s)QC, (v)QC, (i)T, (a)libava, IT (d)iodes", default = "s")
    parser.add_argument("-t", help="2S or PSS", required = False)
    parser.add_argument("-i", help="(p)reirrad, post(i)rrad, or post(a)nn", default = 'pia')
    parser.add_argument("-k", help="(I)RRADSENSOR, (AD)ALL DIODES, (D)IODE, (DH)DIODE_HALF, (DHPS)DIODE_HALF_PSTOP, (DHPSGR)DIODE_HALF_PSTOP_GRCONN, (DQ)DIODE_QUARTER", default = 'I')
    parser.add_argument("-n", help="(n)eutron or (p)roton or (x)ray", required = False)
    parser.add_argument("-v", help="600, 800, all", required = False)
    parser.add_argument("-a", help="Annealing Step, Brown or KIT: (2B) = Brown Ann2, (3K) KIT Ann3", required = False)
    parser.add_argument("-l", help="Location (e.g. Brown, KIT)", required = False)
    parser.add_argument("-r", help="True to pull data from db, False to plot from saved csv", default = False)
    parser.add_argument("-p", default=False, help="Create plots from the sensor data?")
    args = parser.parse_args()
    label = args.g
    grouping = ""
    filters = ""
    if label is None:
        grouping = "Sensor"
    else:
        if "b" in label:
            grouping = "Batch_"
        if "w" in label or "s" in label:
            grouping = "Sensor_"
        if "t" in label:
            grouping += "Flavor_"
        if "n" in label:
            grouping += "Particle_"
        if "k" in label:
            grouping += "Structure_"
        if "a" in label:
            grouping += "Ann_"
        if "d" in label or "D" in label:
            grouping += "Date_"
        if "r" in label:
            grouping += "RunType_"
        grouping = grouping[:-1]

    status = args.i
    irr = ""
    if "p" in status:
        irr = "PreIrr_"
    if "i" in status:
        irr += "PostIrr_"
    if ("a" in status) and (args.m != "d"):
        irr += "PostAnn_"
    if irr != "" and args.m != "c" and args.m != "s":
        irr = irr[:-1]
        filters += "Irrad Status: " + irr + "\n"

    structs = []
    GRConn = False
    if args.k == "I":
        if args.m == "d" or args.m == "i" or args.m == "a":
            structcond = "e.KIND_OF_HM_STRUCT_ID = 'BABYSENSOR'"
            structs.append(["BABYSENSOR", "F"])
            filters += "Structures: BABYSENSOR\n"
        else:
            structs.append(["SENSOR", "F"])
    elif args.k == "AD":
        structcond = "'DIODE' in e.KIND_OF_HM_STRUCT_ID"
        structs = [["DIODE", "F"], ["DIODE_HALF", "F"], ["DIODE_HALF_PSTOP", "F"], ["DIODE_HALF_PSTOP", "T"], ["DIODE_QUARTER", "F"]]
    elif args.k == "D":
        structcond = "e.KIND_OF_HM_STRUCT_ID = 'DIODE'"
        structs.append(["DIODE", "F"])
        filters += "Structures: DIODE\n"
    elif args.k == "DH":
        structcond = "e.KIND_OF_HM_STRUCT_ID = 'DIODE_HALF'"
        structs.append(["DIODE_HALF", "F"])
        filters += "Structures: DIODEHALF\n"
    elif args.k == "DHPS":
        structcond = "e.KIND_OF_HM_STRUCT_ID = 'DIODE_HALF_PSTOP'"
        structs.append(["DIODE_HALF_PSTOP", "F"])
        filters += "Structures: DIODEHALFPSTOP\n"
    elif args.k == "DHPSGR":
        structcond = "e.KIND_OF_HM_STRUCT_ID = 'DIODE_HALF_PSTOP'"
        structs.append(["DIODE_HALF_PSTOP", "T"])
        GRConn = True
        filters += "Structures: DIODEHALFPSTOP_GR\n"
    elif args.k == "DQ":
        structcond = "e.KIND_OF_HM_STRUCT_ID = 'DIODE_QUARTER'"
        structs.append(["DIODE_QUARTER", "F"])
        filters += "Structures: DIODEQUARTER\n"
    #print ("batch", args.b)
    if args.v is not None:
        if '600' in args.v:
            voltages = ['-600']
        elif '800' in args.v:
            voltages = ['-800']
        elif args.v == 'all':
            voltages = ['-600','-800']
    elif args.m == "a":
        voltages = ['-600','-800']
    else:
        voltages = ['SQC_-600']
    if args.m == "c":
        measurements = ['SQC', 'VQC', 'mod_tailEncapsulated']
    elif args.m == "v":
        measurements = ['VQC']
    elif (args.m == "i") or (args.m == "a") or (args.m == "d"):
        measurements = ['IT']
    else:
        measurements = ['SQC']

    conditions = ""
    if "IT" in measurements:
        if args.t == '2S':
            conditions += "and p.KIND_OF_PART = '2S Halfmoon S'"
        elif args.t == 'PSS':
            conditions += "and p.KIND_OF_PART = 'PS-s Halfmoon SW'"
    else:
        if args.t == '2S':
            conditions += "and p.KIND_OF_PART = '2S Sensor'"
        elif args.t == 'PSS':
            conditions += "and p.KIND_OF_PART = 'PS-s Sensor'"

    if args.l is not None:
        conditions += f"and d.LOCATION = '{args.l}'"

    if args.n is not None:
        conditions += f"and e.RADIATION_TYP = '{args.n}'"

    if args.b is not None:
        batches = args.b[0]
        minbatch = min(batches)
        maxbatch = max(batches)
        if minbatch == maxbatch:
            batch = minbatch
        else:
            batch = minbatch + "_" + maxbatch
        print (filters)
        print ("Getting sensors from batch(es)", batch)
        #print (batches, minbatch, maxbatch, batch)
        if args.m == "c":
            query = f"select p.NAME_LABEL from trker_cmsr.c1700 c inner join trker_cmsr.parts p on p.ID = c.PART_ID inner join trker_cmsr.datasets d on d.ID = c.CONDITION_DATA_SET_ID where p.BATCH_NUMBER='{batch}' and d.RUN_TYPE='SQC' and c.volts='-5' order by p.ID"
        elif args.m == "v":
            query = f"select p.NAME_LABEL from trker_cmsr.c1700 c inner join trker_cmsr.parts p on p.ID = c.PART_ID inner join trker_cmsr.datasets d on d.ID = c.CONDITION_DATA_SET_ID where p.BATCH_NUMBER='{batch}' and d.RUN_TYPE='VQC' and c.volts='-20' order by p.ID"
        elif args.m == "i":
            query = f"select p.NAME_LABEL from trker_cmsr.c8780 c inner join trker_cmsr.parts p on p.ID = c.PART_ID inner join trker_cmsr.datasets d on d.ID = c.CONDITION_DATA_SET_ID inner join trker_cmsr.c15400 e on e.CONDITION_DATA_SET_ID = c.AGGREGATED_COND_DATA_SET_ID where p.BATCH_NUMBER>='{minbatch}' and p.BATCH_NUMBER<='{maxbatch}' and d.RUN_TYPE='IT' and c.volts='-10' " + conditions + " order by p.ID"
        elif args.m == "d":
            query = f"select p.NAME_LABEL from trker_cmsr.c8780 c inner join trker_cmsr.parts p on p.ID = c.PART_ID inner join trker_cmsr.datasets d on d.ID = c.CONDITION_DATA_SET_ID inner join trker_cmsr.c15400 e on e.CONDITION_DATA_SET_ID = c.AGGREGATED_COND_DATA_SET_ID where p.BATCH_NUMBER>='{minbatch}' and p.BATCH_NUMBER<='{maxbatch}' and d.RUN_TYPE='IT' and c.volts='-10' " + conditions + " order by p.ID"
        elif args.m == "a":
            query = f"select p.NAME_LABEL from trker_cmsr.c15420 c inner join trker_cmsr.parts p on p.ID = c.PART_ID inner join trker_cmsr.datasets d on d.ID = c.CONDITION_DATA_SET_ID inner join trker_cmsr.c15400 e on e.CONDITION_DATA_SET_ID = c.AGGREGATED_COND_DATA_SET_ID where p.BATCH_NUMBER>='{minbatch}' and p.BATCH_NUMBER<='{maxbatch}' and d.RUN_TYPE='IT' and c.voltage_v='-600' " + conditions + " order by p.ID"
        else:
            query = f"select p.NAME_LABEL from trker_cmsr.c3980 c inner join trker_cmsr.parts p on p.ID = c.PART_ID inner join trker_cmsr.datasets d on d.ID = c.CONDITION_DATA_SET_ID where p.BATCH_NUMBER>='{minbatch}' and p.BATCH_NUMBER<='{maxbatch}' and d.RUN_TYPE='SQC' and c.strip='4' " + conditions + " order by p.ID"

        #print ("query", "{}".format(query))
        initial_path = os.getcwd()
        os.chdir('./resthub/clients/python/src/main/python/')
        p1 = subprocess.run(
            [
                "python3",
                "rhapi.py",
                "-n",
                "--krb",
                "-all",
                "--url=http://dbloader-tracker:8113",
                "{}".format(query),
            ],
                capture_output=True,
        )
        data = p1.stdout.decode().splitlines()
        os.chdir(initial_path)
        sensorlist = data[1:-1]
        sensorlist = np.unique(np.array(sensorlist))
        print (len(sensorlist), " sensors found: ", sensorlist)
    else:
        sensorlist = [args.s]
        batch = sensorlist[0][0:5]
        print ("Sensor:", sensorlist, batch)
    if args.m == "c":
        model = "KindOfMeasurement_CVIV.json"
        if args.s is None:
            folder = "CVIV_" + batch + "_" + grouping
        else:
            folder = "CVIV_" + args.s + "_" + grouping
    if args.m == "v":
        model = "KindOfMeasurement_CVIV.json"
        if args.s is None:
            folder = "VQC_" + batch + "_" + grouping
        else:
            folder = "VQC_" + args.s + "_" + grouping
    elif args.m == "d":
        model = "KindOfMeasurement_IT_Diodes.json"
        folder = "IT_DIODEs_" + batch + "_" + irr + "_" + grouping
    elif args.m == "i":
        model = "KindOfMeasurement_IT.json"
        folder = "IT_" + batch + "_" + irr + "_" + grouping
    elif args.m == "a":
        model = "KindOfMeasurement_Alibava.json"
        folder = "Alibava_" + batch + "_" + irr + "_" + grouping
    else:
        model = "KindOfMeasurement.json"
        if args.s is None:
            folder = "SQC_" + batch + "_" + grouping
        else:
            folder = "SQC_" + args.s + "_" + grouping

    if args.r:
        print ("Getting data from DB")
        for sensor in sensorlist:
           get_meas_files(sensor, model, measurements, structs)
    if args.p:
        create_plots(sensorlist, model, measurements, folder, label, status, voltages, structs)
