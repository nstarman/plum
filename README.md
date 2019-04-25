# [Plum: Multiple Dispatch in Python](https://github.com/wesselb/plum)

[![Build](https://travis-ci.org/wesselb/plum.svg?branch=master)](https://travis-ci.org/wesselb/plum)
[![Coverage Status](https://coveralls.io/repos/github/wesselb/plum/badge.svg?branch=master&service=github)](https://coveralls.io/github/wesselb/plum?branch=master)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://wesselb.github.io/plum)

Everybody likes multiple dispatch, just like everybody likes plums.

## Examples
### Promotion
```python
from plum import dispatch, promote, add_promotion_rule, add_conversion_method

@dispatch(object, object)
def add(x, y):
    return add(*promote(x, y))
    
    
@dispatch(int, int)
def add(x, y):
    return x + y
    
    
@dispatch(float, float)
def add(x, y):
    return x + y
```

```python
>>> add(1, 2)
3

>>> add(1.0, 2.0)
3.0

>>> add(1, 2.0)
TypeError: No promotion rule for "builtins.int" and "builtins.float".

>>> add_promotion_rule(int, float, float)

>>> add(1, 2.0)
TypeError: Cannot convert a "builtins.int" to a "builtins.float".

>>> add_conversion_method(type_from=int, type_to=float, f=float)

>>> add(1, 2.0)
3.0
```

### Return Types
```python
from plum import dispatch, add_conversion_method

@dispatch({int, str}, return_type=int)
def f(x):
    return x
```

```python
>>> f(1)
1

>>> f('1')
TypeError: Expected return type "builtins.int", but got type "builtins.str".

>>> add_conversion_method(type_from=str, type_to=int, f=int)

>>> f('1')
1

```

### Method Precedence
```python
from plum import dispatch

class Element(object):
    pass


class ZeroElement(Element):
    pass


class SpecialisedElement(Element):
    pass


@dispatch(ZeroElement, Element)
def mul_no_precedence(a, b):
    return 'zero'


@dispatch(Element, SpecialisedElement)
def mul_no_precedence(a, b):
    return 'specialised operation'
    

@dispatch(ZeroElement, Element, precedence=1)
def mul(a, b):
    return 'zero'


@dispatch(Element, SpecialisedElement)
def mul(a, b):
    return 'specialised operation'
```

```python
>>> zero = ZeroElement()

>>> specialised_element = SpecialisedElement()

>>> element = Element()

>>> mul(zero, element)
'zero'

>>> mul(element, specialised_element)
'specialised operation'

>>> mul_no_precedence(zero, specialised_element)
AmbiguousLookupError: For function "mul_no_precedence", signature (__main__.ZeroElement, __main__.SpecialisedElement) is ambiguous among the following:
  (__main__.ZeroElement, __main__.Element) (precedence: 0)
  (__main__.Element, __main__.SpecialisedElement) (precedence: 0)

>>> mul(zero, specialised_element)
'zero'
```

### Parametric Classes
```python
from plum import dispatch, parametric

@parametric
class A(object):  # Must be a new-style class!
    pass
    
@dispatch(A)
def f(x):
    return 'fallback'
    
@dispatch(A(1))
def f(x):
    return '1'
    
@dispatch(A(2))
def f(x):
    return '2'
```

```python
>>> A
__main__.A

>>> A(1)
__main__.A{1}

>>> issubclass(A(1), A)
True

>>> A(1)()
<__main__.A{1} at 0x10c2bab70>

>>> f(A(1)())
'1'

>>> f(A(2)())
'2'

>>> f(A(3)())
'fallback'
```


### Variable Types
```python
from plum import dispatch

@dispatch(int)
def f(x):
    print('single argument')

@dispatch(int, [int])
def f(x, *xs):
    print('multiple arguments')
```

```
>>> f(1)
single argument

>>> f(1, 2)
multiple arguments

>>> f(1, 2, 3)
multiple arguments
```

### Union Types
```python
from plum import dispatch

@dispatch(object)
def f(x):
    print('fallback')

@dispatch({int, str})
def f(x):
    print('int or str')
```


```
>>> f(1)
int or str

>>> f('1')
int or str

>>> f(1.0)
fallback
```


### Inheritance
```python
from plum import Dispatcher, Referentiable, Self

class Kernel(Referentiable):
    dispatch = Dispatcher(in_class=Self)

    @dispatch(Self)
    def __add__(self, other):
        return 'kernel'

class StationaryKernel(Kernel, Referentiable):
    dispatch = Dispatcher(in_class=Self)

    @dispatch(Self)
    def __add__(self, other):
        return 'stationary kernel'

kernel = Kernel()
stationary_kernel = StationaryKernel()
```


```
>>> kernel + kernel
'kernel'

>>> kernel + stationary_kernel
'kernel'

>>> stationary_kernel + kernel
'kernel'

>>> stationary_kernel + stationary_kernel
'stationary kernel'
```



