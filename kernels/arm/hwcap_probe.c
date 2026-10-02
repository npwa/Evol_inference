/* Print which Arm ISA extensions the (emulated or real) CPU reports via the ELF auxiliary vector.
 * Used at tier T2 to see which kernel paths QEMU can exercise, and on T3/T4 hardware for provenance. */
#include <stdio.h>
#include <sys/auxv.h>

#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1UL << 20)
#endif
#ifndef HWCAP_SVE
#define HWCAP_SVE (1UL << 22)
#endif
#ifndef HWCAP_ASIMDHP
#define HWCAP_ASIMDHP (1UL << 10)
#endif

int main(void) {
    unsigned long h = getauxval(AT_HWCAP), h2 = getauxval(AT_HWCAP2);
    struct { const char *name; int ok; } f[] = {
        {"asimd (NEON)", !!(h & (1UL << 1))},
        {"asimdhp (fp16 vector)", !!(h & HWCAP_ASIMDHP)},
        {"asimddp (dotprod / SDOT)", !!(h & HWCAP_ASIMDDP)},
        {"sve", !!(h & HWCAP_SVE)},
        {"sve2", !!(h2 & (1UL << 1))},
        {"i8mm (SMMLA/UMMLA)", !!(h2 & (1UL << 13))},
        {"bf16", !!(h2 & (1UL << 14))},
        {"sme", !!(h2 & (1UL << 23))},
        {"sme2", !!(h2 & (1UL << 37))},
    };
    for (unsigned i = 0; i < sizeof f / sizeof f[0]; i++) printf("%-28s %s\n", f[i].name, f[i].ok ? "yes" : "no");
    return 0;
}
