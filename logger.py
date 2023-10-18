# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 19:27:06 2022

@author: czjghost
"""

import sys
import os
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
#        self.log = open(filename, 'a')
        self.filename = filename
 
    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        with open(self.filename, 'a') as log:    
            log.write(message)
         
    def flush(self):
        pass

def build_er_holder(args):
    holder = ""
    if os.path.exists("/data/lgq/"):
        holder = "/data/lgq/"
        
    if args.seed >= 0:
        holder = holder + "result/" + "seed=" + str(args.seed) + "/" 
    else:
        holder = holder + "result/" + "seed=random" + "/"
        
    holder = holder + args.data + "_" + args.classify
    holder = holder + "_" + args.loss + "_" + args.retrieve 
        
    if args.loss == "focal":
        holder = holder + "_α=" + str(args.focal_alpha) + "_γ=" + str(args.focal_gamma)
    elif args.loss == "rfocal":
        holder = holder + "_α=" + str(args.rfocal_alpha) + "_σ=" + str(args.rfocal_sigma) + "_μ=" + str(args.rfocal_miu)
        
    if args.kd_trick:
        holder = holder + "_vkd"
        holder = holder + "_T=" + str(args.T)
        holder = holder + "_prob=" + str(args.cor_prob)
        holder = holder + "_λ=" + str(args.kd_lamda)
    
  
    holder = holder + "_eps=" + str(args.eps_mem_batch)
    holder = holder + "_mem=" + str(args.mem_size)
    holder = holder + "_" + str(args.optimizer)
    holder = holder + "_lr=" + str(args.learning_rate)
    
    if args.review_trick:
        holder = holder + "_rev"   
        holder = holder + "_batch=" + str(args.rv_batch)
    if args.fix_order:
        holder = holder + "_fix"

    if not os.path.exists(holder):
        os.makedirs(holder)
        
    return holder

if __name__ == "__main__":
    
    sys.stdout = Logger(stream=sys.stdout)
#    sys.stderr = Logger(stream=sys.stderr)	
    # now it works
    print('print something')
    print("output")