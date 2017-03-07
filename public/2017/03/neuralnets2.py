#!/usr/bin/env python


# pip install click


from __future__ import print_function
from random import random
import click


def forward_add_gate(x,y):
    return x+y


def forward_multiply_gate(q, z):
    return q*z


F = lambda x,y,z: forward_multiply_gate(forward_add_gate(x,y),z)


@click.command()
@click.option('-z', '--step-size', type=float, default=0.01)
@click.argument('x', type=float, nargs=1)
@click.argument('y', type=float, nargs=1)
@click.argument('z', type=float, nargs=1)
def cli(x, y, z, step_size):
    '''
    Python implementation for "Recursive Case: Circuits with Multiple Gates".
    The circuit implements the mathematical function: `f(x,y,z) = (x+y)*z`.
    Based on the javascript code in the blog by Andrej Karpathy
    http://karpathy.github.io/neuralnets/
    
    Usage:
      analytical_gradient start_x start_y start_z
      
    Examples:
      analytical_gradient 2 3 4
      
      # use the double dash -- so that negative numbers can be specified
      analytical_gradient -- -2 5 -4
    '''
    q = forward_add_gate(x, y)
    f = forward_multiply_gate(q, z)

    # gradient of multiply gate
    derivative_f_wrt_z = q
    derivative_f_wrt_q = z
    
    # gradient of additive gate
    derivative_q_wrt_x = 1
    derivative_q_wrt_y = 1

    # gradient using the chain rule
    derivative_f_wrt_x = derivative_q_wrt_x * derivative_f_wrt_q
    derivative_f_wrt_y = derivative_q_wrt_y * derivative_f_wrt_q

    best_x = x + step_size * derivative_f_wrt_x
    best_y = y + step_size * derivative_f_wrt_y
    best_z = z + step_size * derivative_f_wrt_z
    best_out = F(best_x, best_y, best_z)

    print("Analytical Method:")
    print_result(x, y, z, best_x, best_y, best_z, best_out)    


    # check via numerical gradient
    x_derivative = (F(x+step_size, y, z) - F(x, y, z)) / step_size
    y_derivative = (F(x, y+step_size, z) - F(x, y, z)) / step_size
    z_derivative = (F(x, y, z+step_size) - F(x, y, z)) / step_size
    
    numerical_x = x + step_size * x_derivative
    numerical_y = y + step_size * y_derivative
    numerical_z = z + step_size * z_derivative
    numerical_out = F(numerical_x, numerical_y, numerical_z)

    print("\nNumerical Method:")
    print_result(x, y, z, numerical_x, numerical_y, numerical_z, numerical_out)    

    if round(best_out, 4) == round(numerical_out, 4):
        print("\nSimilar answer, WHEW!")
    else:
        raise RuntimeError("Expected the same answer. :(")

    
def print_result(x, y, z, best_x, best_y, best_z, best_out):
    print("(%.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f) = %.4f" % (
            x, y, z, best_x, best_y, best_z, best_out))

    
if __name__ == '__main__':
    cli()
