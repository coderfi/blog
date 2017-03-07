#!/usr/bin/env python


# pip install click


from __future__ import print_function
from random import random
import click


def forward_multiply_gate(x, y):
    return x*y


F = forward_multiply_gate


@click.group()
def cli():
    '''
    Example python implementations for `f(x,y) = x*y`.
    Based on the javascript code in the blog by Andrej Karpathy
    http://karpathy.github.io/neuralnets/
    '''
        
    pass


@cli.command()
@click.option('-e', '--step-size', type=float, default=0.01)
@click.option('-s', '--steps', type=int, default=1000)
@click.argument('x', type=float, nargs=1)
@click.argument('y', type=float, nargs=1)
def random_local_search(x, y, step_size, steps):
    '''
    Random Local Search example.
    
    Usage:
      random_local_search x y
      
    Examples:
      random_local_search 2 3
      
      # use the double dash -- so that negative numbers can be specified
      random_local_search -- 2 -3
    '''
    best_out = -2**32
    best_x = x
    best_y = y
    
    for _ in xrange(steps):
        x2 = x + step_size * (random() * 2 - 1)
        y2 = y + step_size * (random() * 2 - 1)

        answer = F(x2, y2)
        if answer > best_out:
            best_x = x2
            best_y = y2
            best_out = answer

    print_result(x, y, best_x, best_y, best_out)


@cli.command()
@click.option('-e', '--step-size', type=float, default=0.01)
@click.argument('x', type=float, nargs=1)
@click.argument('y', type=float, nargs=1)
def numerical_gradient(x, y, step_size):
    '''
    Numerical Gradient example.
    
    Usage:
      numerical_gradient start_x start_y
      
    Examples:
      numerical_gradient 2 3
      
      # use the double dash -- so that negative numbers can be specified
      numerical_gradient -- 2 -3
    '''
    out = F(x, y)
    x_derivative = (F(x + step_size, y) - out) / step_size
    y_derivative = (F(x, y + step_size) - out) / step_size

    best_x = x + step_size * x_derivative
    best_y = y + step_size * y_derivative
    best_out = F(best_x, best_y)

    print_result(x, y, best_x, best_y, best_out)


@cli.command()
@click.option('-e', '--step-size', type=float, default=0.01)
@click.argument('x', type=float, nargs=1)
@click.argument('y', type=float, nargs=1)
def analytical_gradient(x, y, step_size):
    '''
    Analytical Gradient example.
    
    Usage:
      analytical_gradient start_x start_y
      
    Examples:
      analytical_gradient 2 3
      
      # use the double dash -- so that negative numbers can be specified
      analytical_gradient -- 2 -3
    '''
    x_gradient = y
    y_gradient = x

    best_x = x + step_size * x_gradient
    best_y = y + step_size * y_gradient
    best_out = F(best_x, best_y)

    print_result(x, y, best_x, best_y, best_out)    


def print_result(x, y, best_x, best_y, best_out):
    print("(%.4f, %.4f) -> (%.4f, %.4f) = %.4f" % (
            x, y, best_x, best_y, best_out))

    
if __name__ == '__main__':
    cli()
