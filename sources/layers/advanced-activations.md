<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L19)</span>
### LeakyReLU

```python
keras.layers.LeakyReLU(alpha=0.3)
```

ReLU(Rectified Linear Unit) 활성화 함수의 leaky version.

유닛이 활성화되지 않는 경우 작은 그래디언트를 허용합니다

LeakyReLU는 다음과 같습니다.  
`f(x) = alpha * x for x < 0`,  
`f(x) = x         for x >= 0`.

__입력 형태__

임의로 설정할 수 있습니다. `LeakyReLU`를 모델의 첫 번째 층으로 사용할 때 키워드 인자 `input_shape`(샘플 축 제외한 `int` 튜플)을 지정해야 합니다.

__출력 형태__

입력 형태와 동일합니다.

__인자__

- __alpha__: 음이 아닌 `float`. 음의 부분 기울기 계수입니다.

__참고__

- [Rectifier Nonlinearities Improve Neural Network Acoustic Models](
   https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L59)</span>
### PReLU

```python
keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)
```

Parametric Rectified Linear Unit 활성화 함수.

PReLU는 다음과 같습니다.  
`f(x) = alpha * x for x < 0`,  
`f(x) = x for x >= 0`,  
이때 `alpha`는 x와 동일한 형태를 가진 학습된 배열입니다.

__입력 형태__

임의로 설정할 수 있습니다. `PReLU`를 모델의 첫 번째 층으로 사용할 때 키워드 인자 `input_shape`(샘플 축을 제외한 `int` 튜플)을 지정해야 합니다.

__출력 형태__

입력 형태와 동일합니다.

__인자__

- __alpha_initializer__: weights를 위한 initializer 함수입니다.
- __alpha_regularizer__: weights를 위한 regularizer 함수입니다.
- __alpha_constraint__: weights의 constraint입니다.
- __shared_axes__: 활성화 함수에 대해 학습 가능한 parameter들을 공유할 축를 의미합니다. 예를 들어, 만일 입력 feature map들이 `(batch, height, width, channels)`의 출력 형태으로 2D convolution으로부터 생성된 것이며, 각 필터가 하나의 매개 변수 세트를 공유하도록 하고 싶은 경우, `shared_axes=[1, 2]`로 설정하십시오.

__참고__

- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on
   ImageNet Classification](https://arxiv.org/abs/1502.01852)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L153)</span>
### ELU

```python
keras.layers.ELU(alpha=1.0)
```

Exponential Linear Unit 활성화 함수.

ELU는 다음과 같습니다.  
`x < 0인 경우 f(x) =  alpha * (exp(x) - 1.)`,  
`x >= 0인 경우 f(x) = x`.

__입력 형태__

임의로 설정할 수 있습니다. `ELU`를 모델의 첫 번째 층으로 사용할 때 키워드 인자 `input_shape`(샘플 축을 제외한 `int` 튜플)을 지정해야 합니다.

__출력 형태__

입력 형태와 동일합니다.

__인자__

- __alpha__: 음의 부분 factor에 대한 값입니다.

__참고__

- [Fast and Accurate Deep Network Learning by Exponential Linear Units
   (ELUs)](https://arxiv.org/abs/1511.07289v1)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L193)</span>
### ThresholdedReLU

```python
keras.layers.ThresholdedReLU(theta=1.0)
```

ThresholdedReLU 활성화 함수.

ThresholdedReLU는 다음과 같습니다.
`x > theta인 경우 f(x) = x`,  
`그렇지 않은 경우 f(x) = 0`.

__입력 형태__

임의로 설정할 수 있습니다. `ELU`를 모델의 첫 번째 층으로 사용할 때 키워드 인자 `input_shape`(샘플 축을 제외한 `int` 튜플)을 지정해야 합니다.

__출력 형태__

입력 형태와 동일합니다.

__인자__

- __theta__: 음이 아닌 `float`. 활성화가 이루어지는 임계값 위치입니다.

__참고__

- [Zero-Bias Autoencoders and the Benefits of Co-Adapting Features](
   https://arxiv.org/abs/1402.3337)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L233)</span>
### Softmax

```python
keras.layers.Softmax(축=-1)
```

Softmax 활성화 함수입니다.

__입력 형태__

임의로 설정할 수 있습니다. `ELU`를 모델의 첫 번째 층으로 사용할 때 키워드 인자 `input_shape`(샘플 축을 제외한 `int` 튜플)을 지정해야 합니다.

__출력 형태__

입력 형태와 동일합니다.

__인자__

- __축__: softmax normalization가 적용되는 축의 정숫값.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L265)</span>
### ReLU

```python
keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0)
```

Rectified Linear Unit 활성화 함수.

기본 인자들을 사용하면 element-wise `max(x, 0)`를 반환합니다.

다른 인자를 사용하면 다음과 같습니다.
`x >= max_value`인 경우 `f(x) = max_value`,  
`threshold <= x < max_value`인 경우 `f(x) = x`,  
그렇지 않은 경우 `f(x) = negative_slope * (x - threshold)`.

__입력 형태__

임의로 설정할 수 있습니다. `ELU`를 모델의 첫 번째 층으로 사용할 때 키워드 인자 `input_shape`(샘플 축을 제외한 `int` 튜플)을 지정해야 합니다.

__출력 형태__

입력 형태와 동일합니다.

__인자__

- __max_value__: 음이 아닌 `float`. 최대 활성화 값을 의미합니다.
- __negative_slope__: 음이 아닌 `float`. 음의 부분 기울기 계수입니다.
- __threshold__: `float`. 임계값이 정해진 활성화를 위한 임계값을 의미합니다.

