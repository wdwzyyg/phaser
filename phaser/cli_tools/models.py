from __future__ import annotations
import math
import typing as t
from pathlib import Path

from pydantic import BaseModel, PrivateAttr, Field, validator, root_validator
from pydantic.main import ModelMetaclass
from pydantic.types import NonNegativeInt, ConstrainedList
from pydantic.typing import get_sub_types, is_none_type
from pydantic.generics import GenericModel
from pydantic.fields import ModelField

Num = t.TypeVar('Num', bound=t.Union[int, float])
T = t.TypeVar('T')
U = t.TypeVar('U')

def _fix_field_allow_none(field: ModelField):
    if field.allow_none:
        return

    sub_fields = [field] if field.sub_fields is None else field.sub_fields
    for sub_field in sub_fields:
        if sub_field.allow_none:
            field.allow_none = True
            break

        for subtype in get_sub_types(sub_field.type_):
            if is_none_type(subtype) or (isinstance(subtype, type) and
                                            issubclass(subtype, WrapperModel) and
                                            subtype.__fields__['__root__'].allow_none):
                field.allow_none = sub_field.allow_none = True
                break
        else:
            continue
        break


class ModelMeta(ModelMetaclass):
    def __new__(cls, name, bases, namespace, **kwargs):
        ty: BaseModel = super(ModelMeta, cls).__new__(cls, name, bases, namespace, **kwargs)
        # workaround for pydantic assuming inner types don't accept None
        for field in ty.__fields__.values():
            _fix_field_allow_none(field)
        return ty


class ModelConfig(BaseModel, metaclass=ModelMeta):
    class Config:
        allow_population_by_field_name = True
        extra = 'forbid'
        allow_mutation = False
        frozen = True

        json_encoders = {
            # encode empty paths as empty string (for passing through to matlab)
            Path: lambda p: "" if p == Path("") else str(p)
        }


class WrapperModel(GenericModel):
    __root__: t.Any
    _type_params: t.Tuple[type, ...] = ()

    @validator('__root__', pre=True)
    def validator(cls, value):
        if isinstance(value, WrapperModel):  # type: ignore
            return value.__root__
        return value

    def __class_getitem__(cls, params: t.Union[type, t.Tuple[type, ...]]) -> type:
        new_cls = t.cast(t.Type[WrapperModel], super(WrapperModel, cls).__class_getitem__(params))
        new_cls._type_params = (*cls._type_params, *params) if isinstance(params, tuple) else (*cls._type_params, params)

        if not new_cls.__concrete__:
            return new_cls

        # workaround for pydantic assuming inner types don't accept None
        _fix_field_allow_none(new_cls.__fields__['__root__'])

        return new_cls

    def __init__(self, val: t.Any = None, **kwargs: t.Any) -> None:
        if val is not None:
            kwargs.update(__root__=val)
        return super().__init__(**kwargs)

    def __repr_str__(self, join_str: str) -> str:
        return repr(self.__root__)

    def dict(self, **kwargs) -> t.Dict[str, t.Any]:
        d = super().dict(**kwargs)
        if isinstance(d['__root__'], dict):
            return d['__root__']
        return d


if t.TYPE_CHECKING:
    InitNonNegativeInt: t.TypeAlias = NonNegativeInt
    InitNum: t.TypeAlias = Num
else:
    InitNonNegativeInt: t.TypeAlias = t.Optional[NonNegativeInt]
    InitNum: t.TypeAlias = t.Optional[Num]


class Range(ModelConfig, GenericModel, t.Generic[Num]):
    start: Num
    end: Num

    # n and step are optional but are set by validation, so we need some TYPE_CHECKING hackery
    n: InitNonNegativeInt = Field(default=None)
    step: InitNum = Field(default=None)

    @root_validator(pre=False, skip_on_failure=True)
    def _validate(cls, values):
        if values.get('step') is None:
            if values.get('n') is not None and values['n'] > 1:
                if not isinstance(values['start'], float) and (values['end'] - values['start']) % (values['n'] - 1):
                    raise ValueError("Range must be evenly divisible by 'n'")
                values['step'] = (values['end'] - values['start']) / (values['n'] - 1)
            else:
                values['step'] = 1.
            if isinstance(values['step'], int):
                values['step'] = int(values['step'])
        else:
            if values.get('n') is not None:
                raise ValueError("Either 'n' and 'step' may be specified, but not both.")

        if values.get('n') is None:
            if values['start'] > values['end']:
                values['n'] = 0
            else:
                values['n'] = 1 + math.ceil((values['end'] - values['start']) / values['step'] - 1e-6)

        return values

    def __len__(self) -> int:
        return self.n

    def __iter__(self) -> t.Iterator[Num]:
        if self.n == 0:
            return
        val: Num = self.start
        for _ in range(self.n-1):
            yield val
            val = t.cast(Num, val + self.step)
        yield self.end


class OptionalRange(ModelConfig, BaseModel):
    start: t.Optional[float]
    end: t.Optional[float]

    step: t.Optional[float]
    n: t.Optional[int]

    @root_validator(pre=False, skip_on_failure=True)
    def _validate(cls, values):
        if values['step'] is None and values['n'] is None:
            raise ValueError("Either 'n' or 'step' must be specified.")
        if values['step'] is not None and values['n'] is not None:
            raise ValueError("Either 'n' and 'step' may be specified, but not both.")

        return values

    def to_range(self, start: float, end: float) -> Range[float]:
        return Range[float].parse_obj({
            'start': self.start or start,
            'end': self.end or end,
            'step': self.step,
            'n': self.n,
        })


class ListNotEmpty(ConstrainedList, t.Generic[T]):
    min_items = 1


class ValueOrRange(WrapperModel, t.Generic[Num]):
    __root__: t.Union[Num, Range[Num]]

    def __len__(self) -> int:
        if isinstance(self.__root__, (int, float)):
            return 1
        return self.__root__.__len__()

    def __iter__(self) -> t.Iterator[Num]:
        if isinstance(self.__root__, (int, float)):
            yield self.__root__
        else:
            yield from self.__root__


class ListOrNone(WrapperModel, t.Generic[T]):
    __root__: t.Union[None, t.List[T]]

    def __len__(self) -> int:
        if self.__root__ is None:
            return 1
        return self.__root__.__len__()

    def __iter__(self) -> t.Iterator[T]:
        if self.__root__ is None:
            return
        yield from self.__root__


class ValueOrList(WrapperModel, t.Generic[T]):
    __root__: t.Union[T, ListNotEmpty[T]]
    _is_T: bool = PrivateAttr()

    def _init_private_attributes(self):
        if isinstance(self._type_params, tuple):
            ty = t.Any if len(self._type_params) == 0 else self._type_params[0]
        else:
            ty = self._type_params

        field = self.__fields__['__root__']
        subfield = field._create_sub_type(type_=ty, name=field.name + '_0')
        subfield.allow_none = field.allow_none
        # check if __root__ parses as T without error
        self._is_T = subfield.validate(self.__root__, {}, loc='')[1] is None

        super()._init_private_attributes()

    def map(self, f: t.Callable[[T], U], ty: t.Type[U]) -> ValueOrList[U]:
        if self._is_T:
            inner = f(t.cast(T, self.__root__))
        else:
            inner = list(map(f, t.cast(t.List[T], self.__root__)))
        return ValueOrList[ty].parse_obj(inner)

    def __len__(self) -> int:
        if self._is_T:
            return 1
        return t.cast(t.List[T], self.__root__).__len__()

    def __iter__(self) -> t.Iterator[T]:
        if self._is_T:
            yield t.cast(T, self.__root__)
        else:
            yield from t.cast(t.List[T], self.__root__)


class ValueListOrNone(WrapperModel, t.Generic[T]):
    __root__: t.Union[None, T, t.List[T]]
    _is_T: bool = PrivateAttr()

    def _init_private_attributes(self):
        if self.__root__ is None:
            self._is_T = False
        else:
            if isinstance(self._type_params, tuple):
                ty = t.Any if len(self._type_params) == 0 else self._type_params[0]
            else:
                ty = self._type_params

            field = self.__fields__['__root__']
            subfield = ModelField(name=field.name, type_=ty, class_validators=None, model_config=field.model_config)
            # check if __root__ parses as T without error
            self._is_T = subfield.validate(self.__root__, {}, loc='')[1] is None
        super()._init_private_attributes()

    def __len__(self) -> int:
        if self.__root__ is None:
            return 0
        if self._is_T:
            return 1
        return t.cast(t.List[T], self.__root__).__len__()

    def __iter__(self) -> t.Iterator[T]:
        if self.__root__ is None:
            return
        if self._is_T:
            yield t.cast(T, self.__root__)
        else:
            yield from t.cast(t.List[T], self.__root__)


class ValueListOrRange(WrapperModel, t.Generic[Num]):
    __root__: t.Union[Num, Range[Num], t.List[Num]]

    def __len__(self) -> int:
        if isinstance(self.__root__, (int, float)):
            return 1
        return self.__root__.__len__()
 
    def __iter__(self) -> t.Iterator[Num]:
        if isinstance(self.__root__, (int, float)):
            yield self.__root__
        else:
            yield from self.__root__
