import matplotlib.pyplot as plt
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--logs",type=str,default="logs")
parser.add_argument("--outname",type=str,default="plots")
parser.add_argument("-smooth",type=bool,default=False,action="store_true")

args = parser.parse_args()

losses={"s0.loss_cls":dict(),
"s0.acc":dict(),
"s0.loss_bbox":dict(),
"s1.loss_cls":dict(),
"s1.acc":dict(),
"s1.loss_bbox":dict(),
"s2.loss_cls":dict(),
"s2.acc":dict(),
"s2.loss_bbox":dict(),
"loss_rpn_cls":dict(),
"loss_rpn_bbox":dict(),
"loss":dict()}

def plot(losses,*args,smooth=True,outname=""):
    for arg in args:
        value_by_epoch = []
        for epoch in sorted(losses[arg].keys()):
            if len(losses[arg][epoch])>0:
                value_by_epoch.append(sum(losses[arg][epoch])/len(losses[arg][epoch]))
            else:
                value_by_epoch.append(value_by_epoch[-1])

        plt.plot(value_by_epoch,label=arg)

        if smooth:
            avg_by_epoch = [sum(value_by_epoch[i + j] for j in range(10))/10 for i in range(len(value_by_epoch)-10)]
            plt.plot(avg_by_epoch,label=arg + " smooth")

    plt.legend(loc="best")
    plt.savefig((outname if outname.endswith("/") else outname+ "/") + "".join(["".join(arg.split(".")) + "_"  for arg in args])  + ".jpg")
    plt.clf()

directory_name = ""
if args.logs.endswith(".log.json" ):
    logfiles = [args.logs]  
else:
    logfiles = [file for file in os.listdir(args.logs) if file.endswith(".log.json")]
    directory_name = args.logs if args.logs.endswith("/") else args.logs + "/"

for logfile in logfiles:
    for line in open(directory_name + logfile,"r").readlines():
        logs = json.loads(line)
        for log in logs:
            if log in losses:
                if logs["epoch"] in losses[log]:
                    losses[log][logs["epoch"]].append(logs[log])
                else:
                    losses[log][logs["epoch"]] = [logs[log]]

plot(losses,"loss_rpn_cls","loss_rpn_bbox",smooth=args.smooth,outname=args.outname)
plot(losses,"s0.loss_cls","s0.loss_bbox",smooth=args.smooth,outname=args.outname)
plot(losses,"s1.loss_cls","s1.loss_bbox",smooth=args.smooth,outname=args.outname)
plot(losses,"s2.loss_cls","s2.loss_bbox",smooth=args.smooth,outname=args.outname)
plot(losses,"loss",smooth=args.smooth,outname=args.outname)
plot(losses,"s0.acc","s1.acc","s2.acc",smooth=args.smooth,outname=args.outname)


