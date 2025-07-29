use std::alloc::{alloc as std_alloc, dealloc as std_dealloc, Layout};
use std::ptr::copy_nonoverlapping;
use std::slice;
use tinyjson::JsonValue;

#[no_mangle]
pub extern "C" fn alloc(size: i32) -> *mut u8 {
    unsafe {
        let layout = Layout::from_size_align(size as usize, 1).unwrap();
        std_alloc(layout)
    }
}

#[no_mangle]
pub extern "C" fn free(ptr: *mut u8, size: i32) {
    unsafe {
        let layout = Layout::from_size_align(size as usize, 1).unwrap();
        std_dealloc(ptr, layout);
    }
}

#[no_mangle]
pub extern "C" fn execute(ptr: *mut u8, len: i32) -> i32 {
    unsafe {
        let input_slice = slice::from_raw_parts(ptr, len as usize);
        let input_str = std::str::from_utf8(input_slice).unwrap();

        let json: JsonValue = input_str.parse().unwrap();

        let mut left = 0.0;
        let mut right = 0.0;

        if let JsonValue::Object(obj) = json {
            if let Some(v) = obj.get("left") {
                if let Some(n) = v.get::<f64>() {
                    left = *n;
                }
            }
            if let Some(v) = obj.get("right") {
                if let Some(n) = v.get::<f64>() {
                    right = *n;
                }
            }
        }

        let result = (left as i64) + (right as i64);

        let output_str = format!("{{\"result\":{}}}", result);
        let output_bytes = output_str.as_bytes();

        let total_size = 4 + output_bytes.len();
        let layout = Layout::from_size_align(total_size, 1).unwrap();
        let out_ptr = std_alloc(layout);

        let len_bytes = (output_bytes.len() as u32).to_le_bytes();
        copy_nonoverlapping(len_bytes.as_ptr(), out_ptr, 4);
        copy_nonoverlapping(output_bytes.as_ptr(), out_ptr.add(4), output_bytes.len());

        out_ptr as i32
    }
}
