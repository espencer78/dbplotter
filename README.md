# dbplotter
usage: querySQC.py [-h] [-b B [B ...]] [-s S] [-l L] [-m M] [-t T] [-i I] [-n N] [-v V] [-r R] [-p P]

optional arguments:  
  -h, --help    show this help message and exit  
  -b            batch name  
  -s            sensor name  
  -g            Plot labels, (b)atch No, (w)afer No, (d)ate, (D)ate + time, (s)ensor name, (t)ype[2S/PSS], (n)/p, (a)nn, (r)un_type  
  -m            (c)VIV, (s)QC, (i)T, (a)libava, IT (d)iodes  
  -t            2S or PSS  
  -i            (p)reirrad, post(i)rrad, or post(a)nn  
  -l            Location (e.g. "Brown" or "KIT")  
  -n            (n)eutron or (p)roton or (x)ray  
  -v            600, 800, all  
  -r            True to pull data from db, False to plot from saved csv  
  -p            Create plots from the sensor data?  

The data pulled from the database are saved in .csv files in the data folder.  The plots made are saved in the plots folder.  For the aggregate data (i.e. strip current vs measured fluence) the plots and the csv of the data are saved in the plots folder.  Some examples are shown below:  

Examples 1: Plot SQC batch

"python querySQC.py -s 46805_003_2-S_MAIN0 -m s -g bwd -r True -p True":  
This will plot SQC and VQC CVIV curves for the sensor 46805_003_2-S_MAIN0.  The legend will be of the form "46805_003_date"  

"python querySQC.py -b 46805 -m c -g bwd -r True -p True":  
This will plot SQC and VQC CVIV curves for all sensors with an SQC IV curve uploaded in batch 46805.  The legend will be of the form "46805_xxx_date"  

"python querySQC.py -b 46805 -m s -g bwd -r True -p True":  
This will plot SQC strip measurements for all sensors with an SQC IV curve uploaded in batch 46805.  The legend will be of the form "46805_xxx_date"  


Examples 2: Plot IT CVIV and Strip data

"python querySQC.py -b 42256  -m i -g wa -r True -p True":  
This will plot strip measurements for all sensors with IT strip measurements in batch 42256.  It will show pre-irrad, post-irrad, and post-ann (after Alibava) measurements. The legend will be of the form "001_annStatus"  

"python querySQC.py -b 42000 43000 -l Brown -m i -i i -g bw -r True -p True":  
This will plot strip measurements for all sensors with IT strip measurements in batches between 42000 and 43000 tested at Brown and will plot only post-irrad measurements. The legend will be of the form "42256_xxx"  

"python querySQC.py -b 35000 50000 -l Brown -m i -i i -g tn -r True -p True":  
This will plot strip measurements for all sensors with IT strip measurements in batches between 35000 and 50000 and will plot only post-irrad measurements. The measurements will be grouped (color coded) by [2S/PSS] and [n/p]. The legend will be of the form "[2S/PSS]_[n/p]"  

Examples 3: Plot Alibava data

"python querySQC.py -b 35000 50000 -m a -g tnv -r True -p True":  
This will plot Alibava data from batches 35000 to 50000 grouped by "[2S/PSS]_[n/p] [-600V/-800V]"


Editing types of plots to be made

In the models directory are json files that determine which plots are to be made.  There are different files corresponding to the different modes (-m options).

(c)VIV: KindOfMeasurement_CVIV.json
(s)QC: KindOfMeasurement.json
(i)T: KindOfMeasurement_IT.json
(a)libava: KindOfMeasurement_Alibava.json
IT (d)iodes: KindOfMeasurement_IT_Diodes.json

You can also do profile plots, plots where you specify the profileplotx (the variable from the DB you want to plot on the x-axis), and profileploty (here you can select if you want the "mean" or "median" of the data to be plotted.   
