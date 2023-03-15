# code bundel hierarchy
1. `framework: library + supporting objects`
2. `library: package + module`
3. `package.directory`
4. `module.py`
5. `def function():`
6. `class Class: def method():

# match-case
`Python 3.10` 부터 사용할 수 있다. 타 언어의 switch-case 문법과 유사하다.
```python
def http_status(status):
    match status:
        case 400:
            return "Bad request"
        case 401 | 403:
            return "Unauthorized"
        case 404:
            return "Not found"
        case _:
            return "Other error"
```

리스트나 튜플로 인자를 동시에 처리할 수 있는 강력함이 있다.
```python
def greeting(message):
    match message:
        case ["hello"]:
            print("Hello!")
        case ["hello", name]:
            print(f"Hello {name}!")
        case ["hello", *names]:
            for name in names:
                print(f"Hello {name}!")
        case _:
            print("nothing")


greeting(["hello"])
greeting(["hello", "John"])
greeting(["hello", "John", "Doe", "MAC"])
```

```Python
for n in range(1, 101):
	match (n % 3, n % 5):
		case (0, 0): print("FizzBuzz")
		case (0, _): print("Fizz")
		case (_, 0): print("Buzz")
		case _: print(n)
```


# :=
바다코끼리 연산자, Walrus operator, Assignment expression

`Python 3.8` 부터 사용할 수 있다. 표현식에 이름을 부여해 재사용할 수 있도록 만들어준다.

아래와 같이 작성하면 `len()`을 두 번 호출하는 문제를 막을 수 있었다.
```Python
a = [1, 2, 3, 4]
n = len(a)
if n > 5: print(f"List is too long ({n} elements, expected <= 5)")
```

이제는 다음과 같이 한 줄에서 처리할 수 있다.
```Python
a = [1, 2, 3, 4]
if (n := len(a)) > 5: print(f"List is too long ({n} elements, expected <= 5)")
```

아래와 같은 혼동을 피하기 위해 괄호로 감싸줄 필요가 있다.
```Python
n := len(a) > 5
```


## 예시
```Python
while chunk := file.read(128):
	process(chunk)
```

```Python
[y := f(x), y**2, y**3]
```


# with
[PEP 343](https://peps.python.org/pep-0343/)

자원을 획득하고 자동으로 반환해야할 때 사용하면 좋다.
```Python
with EXPRESSION [as VARIABLE]:
	BLOCK

with open(f) as file:
	file.do()

# 실행되는 내용
file = open(f).__enter__(self)
file.do()
file.exit__(self, type, value, traceback)
```


# asyncio
