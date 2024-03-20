# dbplotter
usage: querySQC.py [-h] [-b B [B ...]] [-s S] [-l L] [-m M] [-t T] [-i I] [-n N] [-v V] [-r R] [-p P]

optional arguments:
  -h, --help    show this help message and exit
  -b B [B ...]  batch name
  -s S          sensor name
  -l L          Plot labels, (b)atch No, (w)afer No, (d)ate, (D)ate + time, (s)ensor name, (t)ype[2S/PSS], (n)/p,
                (a)nn
  -m M          (c)VIV, (s)QC, (i)T, (a)libava
  -t T          2S or PSS
  -i I          (p)reirrad, post(i)rrad, or post(a)nn
  -n N          (n)eutron or (p)roton or (x)ray
  -v V          600, 800, all
  -r R          True to pull data from db, False to plot from saved csv
  -p P          Create plots from the sensor data?

Example 1: Plot SQC batch

"python querySQC.py -b 46805 -m c -l bwd" This will plot SQC and VQC CVIV curves for all sensors with an SQC IV curve uploaded in batch 46805.  The legend will be of the form "46805_001_date"
"python querySQC.py -b 46805 -m s -l bwd" This will plot SQC strip measurements for all sensors with an SQC IV curve uploaded in batch 46805.  The legend will be of the form "46805_001_date"
"python querySQC.py -s 46805_003_2-S_MAIN0 -m s -l bwd" This will plot SQC and VQC CVIV curves for all sensors with an SQC IV curve uploaded in batch 46805.  The legend will be of the form "46805_001_date"
