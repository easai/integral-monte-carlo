from integral_monte_carlo.integral import run_test
import numpy as np

def test_1(): # 测试一 
    def f(x): return x*x
    run_test(f, 0.0, 1.0, 1.0/3)

def test_2(): # 测试二
    def g(x): return np.cos(x)
    run_test(g, 0.0, np.pi/2, 1.0)

def test_3(): # 测试三
    def h(x): return np.exp(x)
    run_test(h, 0.0, 1.0, np.exp(1.0)-1.0)