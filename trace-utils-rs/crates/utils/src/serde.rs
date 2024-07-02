pub use num_traits::float::FloatCore;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
#[serde(remote = "OrderedFloat")]
pub struct OrderedFloatDef<T: FloatCore>(T);

impl<T: FloatCore> From<OrderedFloatDef<T>> for OrderedFloat<T> {
    fn from(ordered_float: OrderedFloatDef<T>) -> OrderedFloat<T> {
        OrderedFloat::<T>::from(ordered_float.0)
    }
}
