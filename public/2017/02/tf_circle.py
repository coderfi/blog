#!/usr/bin/env python

from __future__ import print_function  
import numpy as np  
import tensorflow as tf

__version__ = "1"


def tf_circle(radius):  
    ''' Calculates the circumference and area of a circle,
    given the specified radius.
    Returns (circumferece, area) as two floats
    '''
    # set up some constants
    pi = tf.constant(np.pi)
    two = tf.constant(2, tf.float32)

    # our first computation graph! 2*pi
    two_pi = tf.multiply(pi, two)

    # define the radius as a Tensorflow constant
    radius = tf.constant(radius, tf.float32)

    # our second computation graph! radius^2
    radius_squared = tf.multiply(radius, radius)

    # the circumference of a circle is 2*pi*r
    circumference = tf.multiply(two_pi, radius)

    # the area is pi*r^2
    area = tf.multiply(pi, radius_squared)

    # start our session and run our circumference and area computation graphs
    with tf.Session() as sess:
        c = sess.run(circumference)
        a = sess.run(area)

    return c, a


if __name__ == '__main__':  
    # prompt for a radius
    r = float(raw_input('radius: '))
    c, a = tf_circle(r)

    print(("For a circle with radius=%s, "
           "the circumference and radius are "
           "approximately %s and %s, respectively") % (r, c, a))
