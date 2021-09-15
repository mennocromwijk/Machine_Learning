# -*- coding: utf-8 -*-

def calc_distance(list1=[],list2=[]):
    '''function that calculates the distance between two lists using the Euclidean distance
    
    Input:
    list1 and list2: lists of values in all dimension, of which the distance has to calculated for
                    list1 and list2 should be of the same length
                    , otherwise you will get an error.'''
    if len(list1)!= len(list2):
        raise Exception("list1 and list2 in calc_distances(method,list1,list2) are not of the same length")
    tot_dist = 0
    for n in range(len(list1)):
        dist = (list1[n]-list2[n])**2
        tot_dist+=dist
        tot_dist = tot_dist**0.5
    return tot_dist

