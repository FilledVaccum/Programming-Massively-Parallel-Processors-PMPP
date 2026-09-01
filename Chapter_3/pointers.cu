#include <stdio.h>

// ============================================================
// POINTERS TUTORIAL - Run this and study the output
// Compile: nvcc -o pointers pointers.cu
// ============================================================

int main() {

    printf("========================================\n");
    printf("PART 1: A variable and its address\n");
    printf("========================================\n");

    int age = 25;

    printf("age        = %d      (the VALUE stored in the variable)\n", age);
    printf("&age       = %p      (the ADDRESS where 'age' lives in RAM)\n", &age);
    printf("sizeof(age)= %lu bytes (how much space 'age' occupies)\n\n", sizeof(age));
    // & means: "give me the address of this variable"
    // Think: age is a house, &age is the house's street address


    printf("========================================\n");
    printf("PART 2: A pointer - a variable that HOLDS an address\n");
    printf("========================================\n");

    int *ptr = &age;  // ptr is a POINTER. It stores the ADDRESS of 'age'.

    printf("ptr        = %p      (the value INSIDE ptr = address of age)\n", ptr);
    printf("&ptr       = %p      (the address of ptr ITSELF - different!)\n", &ptr);
    printf("*ptr       = %d      (go TO the address ptr holds, read the value there)\n", *ptr);
    printf("age        = %d      (same value! because ptr points to age)\n\n", age);
    // ptr  = the address (where to go)
    // *ptr = the value AT that address (what's there)
    // &ptr = where the pointer variable itself lives


    printf("========================================\n");
    printf("PART 3: Changing value THROUGH a pointer\n");
    printf("========================================\n");

    printf("Before: age = %d, *ptr = %d\n", age, *ptr);
    *ptr = 99;  // go to the address ptr holds, write 99 there
    printf("After *ptr = 99:\n");
    printf("  age  = %d   (age changed! because ptr points to age)\n", age);
    printf("  *ptr = %d   (same value, same memory location)\n\n", *ptr);


    printf("========================================\n");
    printf("PART 4: Pointers and arrays - they're almost the same!\n");
    printf("========================================\n");

    int arr[5] = {10, 20, 30, 40, 50};
    int *arrPtr = arr;  // array name IS already an address (no & needed!)

    printf("arr        = %p   (array name = address of first element)\n", arr);
    printf("&arr[0]    = %p   (same address!)\n", &arr[0]);
    printf("arrPtr     = %p   (pointer holds same address)\n\n", arrPtr);

    printf("arr[0]     = %d   (normal array access)\n", arr[0]);
    printf("*arrPtr    = %d   (pointer dereference = same thing)\n", *arrPtr);
    printf("arrPtr[0]  = %d   (pointer with [] = same thing!)\n\n", arrPtr[0]);

    printf("arr[3]     = %d   (normal: 4th element)\n", arr[3]);
    printf("*(arrPtr+3)= %d   (pointer arithmetic: skip 3 ints)\n", *(arrPtr+3));
    printf("arrPtr[3]  = %d   (pointer with [] = same thing)\n\n", arrPtr[3]);


    printf("========================================\n");
    printf("PART 5: Passing pointer to a function (why kernel gets *input)\n");
    printf("========================================\n");

    // When you write: kernel(deviceInput)
    // The kernel receives a COPY of the pointer
    // But both copies point to the SAME data

    int data[3] = {100, 200, 300};
    int *original = data;  // original points to data

    // Simulate what happens when you pass to a function:
    int *copy = original;  // function receives a COPY of the pointer

    printf("original   = %p\n", original);
    printf("copy       = %p   (same address!)\n", copy);
    printf("copy[1]    = %d   (accesses same data)\n\n", copy[1]);

    copy[1] = 999;  // modify through the copy
    printf("After copy[1] = 999:\n");
    printf("  original[1] = %d   (original sees the change!)\n", original[1]);
    printf("  data[1]     = %d   (the actual data changed!)\n\n", data[1]);
    printf(">> The pointer was copied, but the DATA is shared.\n");
    printf(">> This is exactly what happens in a CUDA kernel.\n\n");


    printf("========================================\n");
    printf("PART 6: Why cudaMalloc needs & (double pointer concept)\n");
    printf("========================================\n");

    // Problem: a function needs to CHANGE where a pointer points
    // Solution: pass the ADDRESS of the pointer (&ptr)

    int x = 42;
    int y = 77;
    int *myPtr = &x;  // myPtr points to x

    printf("Before:\n");
    printf("  myPtr points to: %p (which is x, value %d)\n", myPtr, *myPtr);

    // If a function receives just myPtr, it gets a COPY.
    // Changing the copy doesn't affect myPtr.
    int *functionCopy = myPtr;
    functionCopy = &y;  // only changes the COPY!
    printf("\nAfter changing the COPY only:\n");
    printf("  myPtr still points to: %p (still x, value %d)\n", myPtr, *myPtr);
    printf("  functionCopy points to: %p (now y, value %d)\n", functionCopy, *functionCopy);
    printf("  >> The original pointer didn't change!\n");

    // To actually change myPtr, you need its ADDRESS:
    int **ptrToPtr = &myPtr;  // ptrToPtr holds the address of myPtr
    *ptrToPtr = &y;           // go to where myPtr lives, change it to point to y

    printf("\nAfter changing through &myPtr (double pointer):\n");
    printf("  myPtr now points to: %p (now y, value %d)\n", myPtr, *myPtr);
    printf("  >> The original pointer DID change!\n\n");

    printf("  This is what cudaMalloc does:\n");
    printf("  cudaMalloc(&deviceInput, size)\n");
    printf("       ↑ passes address of the pointer\n");
    printf("  cudaMalloc writes a GPU address INTO deviceInput\n");
    printf("  Without &, deviceInput would stay empty/garbage\n\n");


    printf("========================================\n");
    printf("PART 7: Summary cheat sheet\n");
    printf("========================================\n");
    printf("  int a = 5;        // a is a VALUE (5)\n");
    printf("  int *p = &a;      // p is a POINTER (holds address of a)\n");
    printf("  *p                // DEREFERENCE: go to address, read value (5)\n");
    printf("  &a                // ADDRESS-OF: get address of a\n");
    printf("  p[i]              // same as *(p+i): access i-th element\n");
    printf("  int **pp = &p;    // DOUBLE POINTER: holds address of a pointer\n");
    printf("  *pp = new_addr;   // changes where p points\n\n");

    printf("  CUDA usage:\n");
    printf("  int *devPtr;              // pointer (will hold GPU address)\n");
    printf("  cudaMalloc(&devPtr, n);   // & because cudaMalloc MODIFIES devPtr\n");
    printf("  kernel<<<...>>>(devPtr);  // no & because kernel just READS the address\n");
    printf("  kernel receives *input    // * means 'this is a pointer parameter'\n");
    printf("  input[i]                  // access data at that GPU address\n");

    return 0;
}
