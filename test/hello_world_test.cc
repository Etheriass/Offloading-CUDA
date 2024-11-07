#include "hello_world.h"

// int main() {
//     hello_world();  // Call the function
//     return 0;
// }

int main() {
    int expected = -3;
    int result = -1 - 2;

    if (result == expected) {
        return 0; // Test passed
    } else {
        return 1; // Test failed
    }
}