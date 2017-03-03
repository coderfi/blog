#!/usr/bin/env python

from __future__ import print_function  
import numpy as np  
import tensorflow as tf  
import threading

__version__ = "3"


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


class WrapThread(threading.Thread):  
    def __init__(self, sess, op):
        self.sess = sess
        self.op = op
        self._results = None
        threading.Thread.__init__(self)

    def run(self):
        self._results = self.sess.run(self.op)

    def results(self):
        return self._results


def main():  
    # prompt for a radius
    radius = float(raw_input('radius: '))

    r = tf.constant(radius, tf.float32)

    with tf.Session() as sess:
        t_circumference = WrapThread(sess, TfCircle.circumference(r))
        t_area = WrapThread(sess, TfCircle.area(r))
        t_circumference.start()
        t_area.start()

        t_circumference.join()
        t_area.join()

        c = t_circumference.results()
        a = t_area.results()

    print(("For a circle with radius=%s, "
           "the circumference and radius are "
           "approximately %s and %s, respectively") % (radius, c, a))



if __name__ == '__main__':  
    main()


