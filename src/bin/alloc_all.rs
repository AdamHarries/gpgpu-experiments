extern crate opencl;

use opencl::hl;
use opencl::mem::CLBuffer;
use std::fmt;
use std::mem;

fn main() {
    for platform in hl::get_platforms().iter() {
        println!("Platform: {}", platform.name());
        println!("Platform Version: {}", platform.version());
        println!("Vendor:   {}", platform.vendor());
        println!("Profile:  {}", platform.profile());
        println!("Available extensions: {}", platform.extensions());
        println!("Available devices:");
        for device in platform.get_devices().iter() {
            println!("   Name: {}", device.name());
            println!("   Type: {}", device.device_type());
            println!("   Profile: {}", device.profile());
            println!("   Compute Units: {}", device.compute_units());
            println!("   Global Memory Size: {} MB", device.global_mem_size() / (1024 * 1024));
            println!("   Local Memory Size: {} KB", device.local_mem_size() / (1024));
            println!("   Max Alloc Size: {} MB", device.max_mem_alloc_size() / (1024 * 1024));
        }
    }
    println!("\n");
    // now, try to get a GPU device:
    let (device, ctx, queue) = opencl::util::create_compute_context_prefer(opencl::util::PreferedType::GPUPrefered).unwrap();
    println!("Using device: {}", device.name());
    // get the size of the global size, and the max allocation size
    let device_global_size = device.global_mem_size();
    let device_max_alloc_size = device.max_mem_alloc_size();
    // calculate how many allocations we could do into it
    let max_mem_objects = device_global_size/device_max_alloc_size;
    let buffer_len = device_max_alloc_size / mem::size_of::<isize>();
    // allocate a load of memory
    let mut mem_objects = Vec::<CLBuffer<isize>>::new();
    let mut kernel_arg_strings = Vec::<String>::new();
    for i in 0..max_mem_objects {
        println!("Adding buffer {} of size: {} MB",i, device_max_alloc_size/(1024*1024));
        // let newbuffer: CLBuffer<isize> = ctx.create_buffer(buffer_len, opencl::cl::CL_MEM_READ_WRITE);
        // mem_objects.push(newbuffer);
        let arg_str: String = format!("__global int *glb_alloc_obj{}", i);
        kernel_arg_strings.push(arg_str);
        
    }
    println!("Added buffers!");
    // now build the kernel:

    let mut kernel_header = String::from("__kernel void KERNEL(\n ");
    
    let mut first = true;
    for arg_str in kernel_arg_strings {
        if(first){
            first = false;
        }else{
            kernel_header.push_str(",\n ");
        }
        kernel_header.push_str("\t");
        kernel_header.push_str(&arg_str);
    }
    kernel_header.push_str(",\n\tconst int glb_obj_alloc_count\n)\n");
    println!("Kernel: \n {}", kernel_header);
}
