pub type Pixel = u32;
pub type Inch = f32;
pub type Dpi = f32;
pub type Millimeter = f32;
pub type Centimeter = f32;
pub struct Point<P> {
    pub x: P,
    pub y: P,
}

impl<P> Point<P> {
    pub fn new(x: P, y: P) -> Self {
        Point { x, y }
    }

    pub fn get_x(&self) -> &P {
        &self.x
    }

    pub fn get_y(&self) -> &P {
        &self.y
    }

    pub fn get_x_mut(&mut self) -> &mut P {
        &mut self.x
    }

    pub fn get_y_mut(&mut self) -> &mut P {
        &mut self.y
    }

    pub fn set_x(&mut self, x: P) {
        self.x = x;
    }

    pub fn set_y(&mut self, y: P) {
        self.y = y;
    }

    pub fn into_tuple(self) -> (P, P) {
        (self.x, self.y)
    }
}

impl From<(f64, f64)> for Point<f64> {
    fn from((x, y): (f64, f64)) -> Self {
        Point { x, y }
    }
}

impl From<(f32, f32)> for Point<f32> {
    fn from((x, y): (f32, f32)) -> Self {
        Point { x, y }
    }
}

impl From<(u32, u32)> for Point<u32> {
    fn from((x, y): (u32, u32)) -> Self {
        Point { x, y }
    }
}

impl From<(u64, u64)> for Point<u64> {
    fn from((x, y): (u64, u64)) -> Self {
        Point { x, y }
    }
}

pub fn pixel_from_inches(inches: &Inch, dpi: &Dpi) -> Pixel {
    (inches * dpi) as Pixel
}

pub fn pixel_from_mm(mm: &Millimeter, dpi: &Dpi) -> Pixel {
    pixel_from_inches(&(mm / 25.4), dpi)
}

pub fn pixel_from_cm(cm: &Centimeter, dpi: &Dpi) -> Pixel {
    pixel_from_inches(&(cm / 2.54), dpi)
}

pub fn inches_from_pixel(pixel: &Pixel, dpi: &Dpi) -> Inch {
    (*pixel as Inch) / dpi
}

pub fn mm_from_pixel(pixel: &Pixel, dpi: &Dpi) -> Millimeter {
    inches_from_pixel(pixel, dpi) * 25.4
}

pub fn cm_from_pixel(pixel: &Pixel, dpi: &Dpi) -> Centimeter {
    inches_from_pixel(pixel, dpi) * 2.54
}

pub fn dpi_from_pixel_inches(pixel: &Pixel, inches: &Inch) -> Dpi {
    *pixel as Dpi / *inches
}

pub fn dpi_from_pixel_mm(pixel: &Pixel, mm: &Millimeter) -> Dpi {
    let mm = mm / 25.4;
    dpi_from_pixel_inches(pixel, &mm)
}

pub fn dpi_from_pixel_cm(pixel: &Pixel, cm: &Centimeter) -> Dpi {
    let inches = cm / 2.54;
    dpi_from_pixel_inches(pixel, &inches)
}

pub fn pixels_point_from_inch_point(point: &Point<Inch>, dpi: &Dpi) -> Point<Pixel> {
    (
        pixel_from_inches(&(point.x), dpi),
        pixel_from_inches(&(point.y), dpi),
    )
        .into()
}

pub fn pixels_point_from_mm_point(point: &Point<Millimeter>, dpi: &Dpi) -> Point<Pixel> {
    (
        pixel_from_mm(&(point.x), dpi),
        pixel_from_mm(&(point.y), dpi),
    )
        .into()
}

pub fn pixels_point_from_cm_point(point: &Point<Centimeter>, dpi: &Dpi) -> Point<Pixel> {
    (
        pixel_from_cm(&(point.x), &dpi),
        pixel_from_cm(&(point.y), &dpi),
    )
        .into()
}

pub fn inch_point_from_pixel(point: &Point<Pixel>, dpi: &Dpi) -> Point<Inch> {
    (
        inches_from_pixel(&(point.x), &dpi),
        inches_from_pixel(&(point.y), &dpi),
    )
        .into()
}

pub fn mm_point_from_pixel(point: &Point<Pixel>, dpi: &Dpi) -> Point<Millimeter> {
    (
        mm_from_pixel(&(point.x), &dpi),
        mm_from_pixel(&(point.y), &dpi),
    )
        .into()
}

pub fn cm_point_from_pixel(point: Point<Pixel>, dpi: Dpi) -> Point<Centimeter> {
    (
        cm_from_pixel(&(point.x), &dpi),
        cm_from_pixel(&(point.y), &dpi),
    )
        .into()
}
