#!/usr/bin/env python

from __future__ import print_function  
import numpy as np  
import tensorflow as tf

__version__ = "2"


class TfCircle(object):  
    ''' Helper to create the tensorflow computation graph
    for calculations about a circle.
    '''

    PI = tf.constant(np.pi)
    TWO = tf.constant(2, tf.float32)
    TWO_PI = tf.multiply(PI, TWO)

    @staticmethod
    def circumference(r):
        ''' returns the CG of the circumference of a circle, e.g. 2*pi*r '''
        c = tf.multiply(TfCircle.TWO_PI, r)

        return c

    @staticmethod
    def area(r):
        ''' returns the CG of the area of a circle, e.g. pi*r^2 '''    
        r_squared = tf.multiply(r, r)
        a = tf.multiply(TfCircle.PI, r_squared)

        return a


def main():  
    # prompt for a radius
    radius = float(raw_input('radius: '))

    # create our computation graphs
    r = tf.constant(radius, tf.float32)
    circumference = TfCircle.circumference(r)
    area = TfCircle.area(r)

    # start our session and run our calculations
    with tf.Session() as sess:
        c = sess.run(circumference)
        a = sess.run(area)

    print(("For a circle with radius=%s, "
           "the circumference and radius are "
           "approximately %s and %s, respectively") % (radius, c, a))


if __name__ == '__main__':  
    main()

