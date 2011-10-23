""" Codegen for mtxm

A   I
  +-------------+
K | a b ....... | 
  |     ...     |
  +-------------+

B   J
  +-------------+
K | i j k l ... | 
  |     ...     |
  +-------------+

C   J
  +-------------+
I | w x y z ... | 
  |     ...     |
  +-------------+

# Complex Complex
w += a i - b j
x += a j + b i
y += a k - b l
z += a l + b k

temps: _c += _a _b +- _ai _br


# Real Complex
w += a i
x += a j
y += a k
z += a l

temps: _c += _a _b ; doubled dimj


# Complex Real
w += a i
x += b i
y += a j
z += b j

temps: _c += _az _bz 


# Real Real
w += a i
x += a j
y += a k
z += a l

temps: _c += _a _b
"""

from itertools import product
import logging
logger = logging.getLogger(__name__)

class MTXMGen:
    def __init__(self, cxa=False, cxb=False):
        self.indent = 4
        self.__in_main_loop = False
        self._mask = False
        self._odds = [1]
        self.have_bgp = False
        # Tile i loop in jik loop order
        self.tile_i = False
        self.complex_a = cxa 
        self.complex_b = cxb
        self.complex_dup_cast = ''

    @property
    def complex_c(self):
        return self.complex_a or self.complex_b

    @property
    def complex_complex(self):
        return self.complex_a and self.complex_b

    @property
    def real_complex(self):
        return not self.complex_a and self.complex_b

    @property
    def complex_real(self):
        return self.complex_a and not self.complex_b

    @property
    def real_real(self):
        return not self.complex_a

    def set_complex(self, a, b):
        self.complex_a = a
        self.complex_b = b

    def _temp(self, prefix, x, y):
        return prefix + '_' + str(x) + '_' + str(y)

    def _temps(self, prefix, x, y, size):
        """
        >>> list(MTXMGen()._temps('_x', 'i', 'j', {'i':2, 'j':3}))
        ['_x_0_0', '_x_0_1', '_x_0_2', '_x_1_0', '_x_1_1', '_x_1_2']
        """
        return [self._temp(prefix, i, j) for i, j in product(range(size[x]), range(size[y]))]

    def _post_process(self, lines):
        return lines

    def _header(self, func_name):
        f = lambda x: x and "complex" or ""
        ret = ["void " + func_name + """(long dimi, long dimj, long dimk, double {} * __restrict__ c_x, const double {} * __restrict__ a_x, const double {} * __restrict__ b_x) {{
    int i, j, k, ii;
    double * __restrict__ c = (double*)c_x;
    const double * __restrict__ a = (double*)a_x;
    const double * __restrict__ b = (double*)b_x;""".format(f(self.complex_c), f(self.complex_a), f(self.complex_b))]
        return ret

    def _temp_dec(self, size):
        ret = []
        indent = '    '
        if self.tile_i:
            ret.append("    double* __restrict__ cc = c;")
            ret.append("    const double* __restrict__ bb = b;")
        x = indent + self.vector_type + ' '
        x += ', '.join(self._temps('_c', 'i', 'j', size) +
                self._temps('_b', 'k', 'j', size)) + ';'
        ret.append(x)
        x = indent + self.splat_type + ' ' + ', '.join(self._temps('_a', 'k', 'i', size)) + ';'
        ret.append(x)
        if self.complex_complex:
            if not self.have_bgp:
                x = "{} {} {};".format(indent, self.vector_type, ', '.join(self._temps('_br', 'k', 'j', size)))
                ret.append(x)
            x = "{} {} {};".format(indent, self.splat_type, ', '.join(self._temps('_ai', 'k', 'i', size)))
            ret.append(x)
        elif self.complex_real:
            x = "{} {} {};".format(indent, self.vector_type, ', '.join(self._temps('_az', 'k', 'i', size)))
            ret.append(x)
            x = "{} {} {};".format(indent, self.splat_type, ', '.join(self._temps('_bz', 'k', 'j', size)))
            ret.append(x)
        elif self.real_complex:
            pass
        return ret

    def _extra(self):
        return []

    def _temps_to_load(self, unrolls, z, x, y, tname=None):
        if not tname:
            tname = '_'+z
        ret = []
        ystep = 1
        if y == 'j':
            ystep = self.vector_length
        for i, j in product(range(unrolls[x]), range(0, unrolls[y], ystep)):
            ret.append((self._temp(tname, i, j), i, j))
        return ret

    def _load_a(self, unrolls, indent):
        spaces = ' ' * (self.indent*indent)
        ret = []
        for temp, k, i in self._temps_to_load(unrolls, 'a', 'k', 'i'):
            addr = '(pa+' + str((self.complex_a and 2 or 1)*i) + ')'
            if self.complex_real:
                ret.append(self._load_az(spaces, addr, temp, k, i))
            else:
                ret.append(spaces + temp + ' = {}({});'.format(self.splat_op, addr))
                if self.complex_complex:
                    ret.append(spaces + self._temp('_ai', k, i) + ' = {}({}+1);'.format(self.splat_op, addr))
        return ret

    def _load_b(self, unrolls, indent):
        spaces = ' ' * (self.indent*indent)
        ret = []
        for temp, k, j in self._temps_to_load(unrolls, 'b', 'k', 'j'):
            addr = '(pb+{})'.format(j // (self.complex_real and 2 or 1))
            if self.complex_real:
                ret.append(self._load_bz(spaces, addr, temp, k, j))
            else:
                ret.append(spaces + temp + ' = ' + self.vector_load + addr + ';')
                if self.complex_complex and not self.have_bgp and not self.have_bgq:
                    ret.append(self._load_br(spaces, addr, temp, k, j))
        return ret

    def _load_c(self, unrolls, indent):
        spaces = ' ' * (self.indent*indent)
        ret = []
        for temp, i, j in self._temps_to_load(unrolls, 'c', 'i', 'j'):
            ret.append(spaces + temp + ' = ' + self.vector_zero + ';')
        return ret

    def _load_br(self, spaces, addr, temp, k, j):
        return spaces + self._temp('_br', k, j) + ' = {}({});'.format(self.complex_reverse_dup, addr)

    def _load_az(self, spaces, addr, temp, k, i):
        return spaces + self._temp('_az', k, i) + ' = {}({}{});'.format(self.complex_dup, self.complex_dup_cast, addr)

    def _load_bz(self, spaces, addr, temp, k, j):
        return spaces + self._temp('_bz', k, j) + ' = {}({});'.format(self.pair_splat, addr)

    def _fma(self, at, bt, ct):
        raise NotImplementedError()

    def _fmaddsub(self, at, bt, ct):
        raise NotImplementedError()

    def _maths(self, unrolls, indent=0):
        spaces = ' ' * (self.indent*indent)
        ret = []
        for j, i, k in product(range(0, unrolls['j'], self.vector_length), range(unrolls['i']), range(unrolls['k'])):
            if self.real_real or self.real_complex:
                at = self._temp('_a', k, i)
                bt = self._temp('_b', k, j)
                ct = self._temp('_c', i, j)
                ret.append(spaces + self._fma(at, bt, ct))
            elif self.complex_real:
                at = self._temp('_az', k, i)
                bt = self._temp('_bz', k, j)
                ct = self._temp('_c', i, j)
                ret.append(spaces + self._fma(at, bt, ct))
            elif self.complex_complex:
                at = self._temp('_a', k, i)
                bt = self._temp('_b', k, j)
                ct = self._temp('_c', i, j)
                ret.append(spaces + self._fma(at, bt, ct))
                at = self._temp('_ai', k, i)
                if not self.have_bgp:
                    bt = self._temp('_br', k, j)
                ret.append(spaces + self._fmaddsub(at, bt, ct))
        return ret

    def _array(self, z, x, xx, y, yy, cpx):
        return z + '+(' + x + '+' + xx + ')*dim' + y + (cpx and "*2" or "") + '+' + yy

    def _store_c(self, unrolls, indent, bc_mod=""):
        spaces = ' ' * (self.indent*indent)
        ret = []
        jstep = self.vector_length
        for i, j in product(range(unrolls['i']), range(0, unrolls['j'], jstep)):
            if j + jstep < unrolls['j'] or self.__in_main_loop or not self._mask:    
                ret.append(spaces + '{}('.format(self.vector_store) + self._array(bc_mod+'c', 'i', str(i), 'j', str(j), self.complex_c) + ', ' + self._temp('_' + 'c', i, j) + ');')
            else:
                # This is somewhat AVX specific, but no other arch's currently support masking, so ok.
                ret.append(spaces + '{}('.format(self.mask_store) + self._array(bc_mod+'c', 'i', str(i), 'j', str(j), self.complex_c) + ', mask, ' + self._temp('_' + 'c', i, j) + ');')
        return ret

    def _loops(self, i, size, bc_mod=""):
        if i == 'i':
            start = 'i=0'
            if self.tile_i:
                start = 'i=ii'
            #FIXME Don't include _odds if i%2==0 and only evens
            loops = [size[i]]
            if loops[-1] != 1:
                loops += self._odds
            if self.have_bgp:
                loops = range(size[i], 0, -2)
            for loop in loops:
                i_cond = ""
                if self.tile_i:
                    i_cond = " && i+{0}<=ii+{1}".format(loop, self.tile_i_size)
                yield ('for ({0}; i+{1}<=dimi {2}; i+={1}) {{'.format(start, loop, i_cond), loop)
                start = ''
        elif i == 'j':
            loop = size[i] // (self.complex_c and 2 or 1)
            self.__in_main_loop = True
            yield ("for (j=dimj; j>{0}; j-={0},{1}c+={0}{2},{1}b+={0}{3}) {{".format(loop, bc_mod, self.complex_c and "*2" or "", self.complex_b and "*2" or ""), size[i])
            self.__in_main_loop = False
            start = ''
            for loop in range(size[i]-self.vector_length, 0, -self.vector_length):
                yield (start + "if (j>{}) {{".format(loop//(self.complex_c and 2 or 1)), loop+self.vector_length)
                start = 'else '
            if size[i] == self.vector_length:
                yield ("{", self.vector_length)
            else:
                yield ("else {", self.vector_length)
        elif i == 'k':
            assert(size[i] == 1)
            yield ("for (k=0; k<dimk; k+=1,pb+=dimj{},pa+=dimi{}) {{".format(self.complex_b and "*2" or "", self.complex_a and "*2" or ""), 1)

    def _close_braces(self, indent=0):
        ret = []
        for i in range(indent, -1, -1):
            ret += [' '*(self.indent*i) + '}']
        return ret

    def _inner_loops(self, perm, sizes, indent=0, unrolls=None, bc_mod=""):
        indent += 1
        if not unrolls:
            unrolls = {x:0 for x in perm}
        ret = []
        spaces = ' '*(self.indent*indent)

        if perm == ['k']:
            ret.append(spaces + "const double* __restrict__ pb = {}b;".format(bc_mod))
            ret.append(spaces + "const double* __restrict__ pa = a+i{};".format(self.complex_a and "*2" or ""))
            ret += self._load_c(unrolls, indent)

        if perm == ['j', 'k']:
            bc_mod = "x"
            ret.append(spaces + "const double* __restrict__ {}b = b;".format(bc_mod))
            ret.append(spaces + "double* __restrict__ {}c = c;".format(bc_mod))

        for loop, unroll in self._loops(perm[0], sizes, bc_mod):
            unrolls[perm[0]] = unroll
            ret.append(spaces + loop)
            if len(perm) > 1:
                ret += self._inner_loops(perm[1:], sizes, indent, unrolls, bc_mod)
            else:
                ret += self._load_a(unrolls, indent+1)
                b_loads = self._load_b(unrolls, indent+1)
                maths = self._maths(unrolls, indent+1)
                b_take = (self.complex_complex and not self.have_bgp) and 2 or 1
                m_take = unrolls['i']*(self.complex_complex and 2 or 1)
                while b_loads:
                    ret += b_loads[0:b_take]
                    ret += maths[0:m_take]
                    b_loads = b_loads[b_take:]
                    maths = maths[m_take:]
            ret.append(spaces + '}')

        if perm == ['k']:
            ret += self._store_c(unrolls, indent, bc_mod)

        return ret

    def _i_tile_loop(self):
        ret = []
        ret.append("    for (ii=0; ii<dimi; ii+={}) {{".format(self.tile_i_size))
        ret.append("        b = bb;")
        ret.append("        c = cc;")
        return ret


    def gen(self, f, perm, size, itile=0, func_name='mtxmq'):
        """Output generated code to file f

        Input:
            perm - an array of 'i', 'j', 'k' in the desired loop order
            size - { index : int, }

        Output:
            None
            Code printed to file f
        """

        if type(perm) is not list:
            perm = list(perm)

        if perm[-1] != 'k':
            raise Exception("k must be inner loop")

        indent = 0
        if itile > 0:
            self.tile_i = True
            self.tile_i_size = itile
            indent += 1

        lines = []

        # Header
        lines += self._header(func_name)

        # Temps Declaration
        lines += self._temp_dec(size)

        # Architecture Specific declarations, e.g. mask prep
        lines += self._extra()

        if self.tile_i:
            lines += self._i_tile_loop()

        # Computation
        lines += self._inner_loops(perm, size, indent)

        # Close braces
        lines += self._close_braces(indent)

        lines = self._post_process(lines)

        # Output
        for line in lines:
            print(line, file=f)


class MTXMAVX(MTXMGen):
    def __init__(self, *args):
        super().__init__(*args)
        self.vector_length = 4

        self.vector_type = '__m256d'
        self.vector_load = '_mm256_loadu_pd'
        self.vector_store = '_mm256_storeu_pd'
        self.vector_zero = '_mm256_setzero_pd()'

        self._mask = True
        self.mask_store = '_mm256_maskstore_pd'

        self.splat_type = '__m256d'
        self.splat_op = '_mm256_broadcast_sd'

        #self.complex_reverse_dup = '_mm256_permute_pd' # (_mm256_loadu_pd(addr), 5), could also use shuffle 5
        self.complex_dup = '_mm256_broadcast_pd'
        self.complex_dup_cast = '(const __m128d*)'
        #self.pair_splat = ''

    def _load_bz(self, spaces, addr, temp, k, j):
        return spaces + self._temp('_bz', k, j) + ' = _mm256_permute_pd(_mm256_broadcast_pd((const __m128d*){}),12);'.format(addr)

    def _load_br(self, spaces, addr, temp, k, j):
        return spaces + self._temp('_br', k, j) + ' = _mm256_permute_pd({}, 5);'.format(temp)

    def _fma(self, at, bt, ct):
        return ct + ' = _mm256_add_pd(_mm256_mul_pd(' + bt + ', ' + at + '), ' + ct + ');'

    def _fmaddsub(self, at, bt, ct):
        return ct + ' = _mm256_addsub_pd(' + ct + ', _mm256_mul_pd(' + at + ', ' + bt + '));'

    def _extra(self):
        if self.real_real:
            return [' ' * self.indent + """
    __m256i mask;
    j = dimj % 4;
    switch (j) {
        case 0:
            mask = _mm256_set_epi32(-1,-1,-1,-1,-1,-1,-1,-1);
            break;
        case 1:
            mask = _mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1);
            break;
        case 2:
            mask = _mm256_set_epi32( 0, 0, 0, 0,-1,-1,-1,-1);
            break;
        case 3:
            mask = _mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1);
            break;
        default:
            return;
    }"""]
        else:
            return [' ' * self.indent + """
    __m256i mask;
    j = dimj % 2;
    switch (j) {
        case 0:
            mask = _mm256_set_epi32(-1,-1,-1,-1,-1,-1,-1,-1);
            break;
        case 1:
            mask = _mm256_set_epi32( 0, 0, 0, 0,-1,-1,-1,-1);
            break;
        default:
            return;
    }"""]

class MTXMSSE(MTXMGen):
    def __init__(self, *args):
        super().__init__(*args)
        self.vector_length = 2

        self.vector_type = '__m128d'
        self.vector_load = '_mm_loadu_pd'
        self.vector_store = '_mm_storeu_pd'
        self.vector_zero = '_mm_setzero_pd()'

        self.splat_type = '__m128d'
        self.splat_op = '_mm_load1_pd'

        self.complex_reverse_dup = '_mm_loadr_pd' # aligned only!
        self.complex_dup = '_mm_loadu_pd'
        self.pair_splat = '_mm_load1_pd'

    def _fma(self, at, bt, ct):
        return ct + ' = _mm_add_pd(_mm_mul_pd(' + bt + ', ' + at + '), ' + ct + ');'

    def _fmaddsub(self, at, bt, ct):
        return "{2} = _mm_addsub_pd({2}, _mm_mul_pd({0}, {1}));".format(at, bt, ct)


class MTXMBGP(MTXMGen):
    def __init__(self, *args):
        super().__init__(*args)
        self.have_bgp = True
        self.vector_length = 2

        self.vector_type = '__complex__ double'
        self.vector_load = '__lfpd'
        self.vector_store = '__stfpd'
        self.vector_zero = '__cmplx(0.0,0.0)'

        self.splat_type = 'double'
        self.splat_op = '*'

        self.complex_reverse_dup = '__lfxd'
        self.complex_dup = '__lfpd'
        self.pair_splat = '*'

    def _fma(self, at, bt, ct):
        return ct + ' = __fxcpmadd(' + ct + ', ' + bt + ', ' + at + ');'

    def _fmaddsub(self, at, bt, ct):
        return ct + ' = __fxcxnpma(' + ct + ', ' + bt + ', ' + at + ');'

    def _post_process(self, lines):
        return [x.replace("__restrict__", "").replace("const", "").replace("double complex", "__complex__ double") for x in lines]

class MTXMBGQ(MTXMGen):
    def __init__(self, *args):
        super().__init__(*args)
        self.have_bgq = True
        self.vector_length = 4

        self.vector_type = 'vector4double'
        self.vector_load = 'vec_ld'
        self.vector_store = 'vec_st'
        self.vector_zero = 'vec_splats(0.0)'

        self.splat_type = 'double'
        self.splat_op = 'vec_splats'

    def _fma(self, at, bt, ct):
        return ct + ' = vec_madd(' + at + ', ' + bt + ', ' + ct + ');'

    def _fmaddsub(self, at, bt, ct):
        return ct + ' = vec_xxnpmadd(' + bt + ', ' + at + ', ' + ct + ');' # this is close but not correct

