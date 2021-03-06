??
?5?5
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	??
?
	ApplyAdam
var"T?	
m"T?	
v"T?
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T?" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
?
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T?

value"T

output_ref"T?"	
Ttype"
validate_shapebool("
use_lockingbool(?
k
BatchMatMulV2
x"T
y"T
output"T"
Ttype:

2	"
adj_xbool( "
adj_ybool( 
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

B
Equal
x"T
y"T
z
"
Ttype:
2	
?
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
,
Floor
x"T
y"T"
Ttype:
2
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
i
LinSpace

start"T	
stop"T
num"Tidx
output"T"
Ttype:
2"
Tidxtype0:
2	
?

LogSoftmax
logits"T

logsoftmax"T"
Ttype:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
?
MaxPoolGrad

orig_input"T
orig_output"T	
grad"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype0:
2	
8
Maximum
x"T
y"T
z"T"
Ttype:

2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
8
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	?
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
j
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
2
StopGradient

input"T
output"T"	
Ttype
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
s

VariableV2
ref"dtype?"
shapeshape"
dtypetype"
	containerstring "
shared_namestring ?
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.14.02unknown??
f
xPlaceholder*
dtype0*(
_output_shapes
:??????????*
shape:??????????
e
y_Placeholder*
dtype0*'
_output_shapes
:?????????
*
shape:?????????

N
	keep_probPlaceholder*
_output_shapes
:*
shape:*
dtype0
g
truncated_normal/shapeConst*
valueB"     *
dtype0*
_output_shapes
:
Z
truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
\
truncated_normal/stddevConst*
valueB
 *???=*
dtype0*
_output_shapes
: 
?
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
T0*
dtype0*
_output_shapes
:	?*
seed2 *

seed 
?
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
_output_shapes
:	?*
T0
n
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*
_output_shapes
:	?
~
Variable
VariableV2*
shape:	?*
shared_name *
dtype0*
_output_shapes
:	?*
	container 
?
Variable/AssignAssignVariabletruncated_normal*
validate_shape(*
_output_shapes
:	?*
use_locking(*
T0*
_class
loc:@Variable
j
Variable/readIdentityVariable*
_class
loc:@Variable*
_output_shapes
:	?*
T0
R
ConstConst*
valueB*???=*
dtype0*
_output_shapes
:
v

Variable_1
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
?
Variable_1/AssignAssign
Variable_1Const*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes
:
k
Variable_1/readIdentity
Variable_1*
_class
loc:@Variable_1*
_output_shapes
:*
T0
z
MatMulMatMulxVariable/read*
T0*'
_output_shapes
:?????????*
transpose_a( *
transpose_b( 
U
addAddMatMulVariable_1/read*
T0*'
_output_shapes
:?????????
C
TanhTanhadd*
T0*'
_output_shapes
:?????????
J
sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
subSubsub/x	keep_prob*
_output_shapes
:*
T0
Q
dropout/ShapeShapeTanh*
T0*
out_type0*
_output_shapes
:
_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
_
dropout/random_uniform/maxConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*

seed *
T0*
dtype0*'
_output_shapes
:?????????*
seed2 
z
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
_output_shapes
: *
T0
?
dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*
T0*'
_output_shapes
:?????????
?
dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*
T0*'
_output_shapes
:?????????
R
dropout/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
I
dropout/subSubdropout/sub/xsub*
T0*
_output_shapes
:
V
dropout/truediv/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
]
dropout/truedivRealDivdropout/truediv/xdropout/sub*
_output_shapes
:*
T0
d
dropout/GreaterEqualGreaterEqualdropout/random_uniformsub*
T0*
_output_shapes
:
L
dropout/mulMulTanhdropout/truediv*
_output_shapes
:*
T0
l
dropout/CastCastdropout/GreaterEqual*

SrcT0
*
Truncate( *
_output_shapes
:*

DstT0
a
dropout/mul_1Muldropout/muldropout/Cast*
T0*'
_output_shapes
:?????????
i
truncated_normal_1/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
\
truncated_normal_1/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
^
truncated_normal_1/stddevConst*
valueB
 *???=*
dtype0*
_output_shapes
: 
?
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*

seed *
T0*
dtype0*
_output_shapes

:*
seed2 
?
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
_output_shapes

:*
T0
s
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
_output_shapes

:*
T0
~

Variable_2
VariableV2*
shape
:*
shared_name *
dtype0*
_output_shapes

:*
	container 
?
Variable_2/AssignAssign
Variable_2truncated_normal_1*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*
_class
loc:@Variable_2
o
Variable_2/readIdentity
Variable_2*
_output_shapes

:*
T0*
_class
loc:@Variable_2
x
b_fc_loc2/initial_valueConst*
dtype0*
_output_shapes
:*-
value$B""  ??              ??    
u
	b_fc_loc2
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
?
b_fc_loc2/AssignAssign	b_fc_loc2b_fc_loc2/initial_value*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@b_fc_loc2*
validate_shape(
h
b_fc_loc2/readIdentity	b_fc_loc2*
_class
loc:@b_fc_loc2*
_output_shapes
:*
T0
?
MatMul_1MatMuldropout/mul_1Variable_2/read*
T0*'
_output_shapes
:?????????*
transpose_a( *
transpose_b( 
X
add_1AddMatMul_1b_fc_loc2/read*'
_output_shapes
:?????????*
T0
G
Tanh_1Tanhadd_1*'
_output_shapes
:?????????*
T0
f
Reshape/shapeConst*
dtype0*
_output_shapes
:*%
valueB"????         
l
ReshapeReshapexReshape/shape*
T0*
Tshape0*/
_output_shapes
:?????????
j
#SpatialTransformer/_transform/ShapeShapeReshape*
T0*
out_type0*
_output_shapes
:
{
1SpatialTransformer/_transform/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
}
3SpatialTransformer/_transform/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
}
3SpatialTransformer/_transform/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
+SpatialTransformer/_transform/strided_sliceStridedSlice#SpatialTransformer/_transform/Shape1SpatialTransformer/_transform/strided_slice/stack3SpatialTransformer/_transform/strided_slice/stack_13SpatialTransformer/_transform/strided_slice/stack_2*
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask 
l
%SpatialTransformer/_transform/Shape_1ShapeReshape*
T0*
out_type0*
_output_shapes
:
}
3SpatialTransformer/_transform/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:

5SpatialTransformer/_transform/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

5SpatialTransformer/_transform/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
-SpatialTransformer/_transform/strided_slice_1StridedSlice%SpatialTransformer/_transform/Shape_13SpatialTransformer/_transform/strided_slice_1/stack5SpatialTransformer/_transform/strided_slice_1/stack_15SpatialTransformer/_transform/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
l
%SpatialTransformer/_transform/Shape_2ShapeReshape*
T0*
out_type0*
_output_shapes
:
}
3SpatialTransformer/_transform/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:

5SpatialTransformer/_transform/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:

5SpatialTransformer/_transform/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
-SpatialTransformer/_transform/strided_slice_2StridedSlice%SpatialTransformer/_transform/Shape_23SpatialTransformer/_transform/strided_slice_2/stack5SpatialTransformer/_transform/strided_slice_2/stack_15SpatialTransformer/_transform/strided_slice_2/stack_2*
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
l
%SpatialTransformer/_transform/Shape_3ShapeReshape*
_output_shapes
:*
T0*
out_type0
}
3SpatialTransformer/_transform/strided_slice_3/stackConst*
valueB:*
dtype0*
_output_shapes
:

5SpatialTransformer/_transform/strided_slice_3/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

5SpatialTransformer/_transform/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
-SpatialTransformer/_transform/strided_slice_3StridedSlice%SpatialTransformer/_transform/Shape_33SpatialTransformer/_transform/strided_slice_3/stack5SpatialTransformer/_transform/strided_slice_3/stack_15SpatialTransformer/_transform/strided_slice_3/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
?
+SpatialTransformer/_transform/Reshape/shapeConst*!
valueB"????      *
dtype0*
_output_shapes
:
?
%SpatialTransformer/_transform/ReshapeReshapeTanh_1+SpatialTransformer/_transform/Reshape/shape*
T0*
Tshape0*+
_output_shapes
:?????????
?
"SpatialTransformer/_transform/CastCast-SpatialTransformer/_transform/strided_slice_1*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
?
$SpatialTransformer/_transform/Cast_1Cast-SpatialTransformer/_transform/strided_slice_2*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
~
-SpatialTransformer/_transform/_meshgrid/stackConst*
valueB"      *
dtype0*
_output_shapes
:
w
2SpatialTransformer/_transform/_meshgrid/ones/ConstConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
,SpatialTransformer/_transform/_meshgrid/onesFill-SpatialTransformer/_transform/_meshgrid/stack2SpatialTransformer/_transform/_meshgrid/ones/Const*
_output_shapes

:*
T0*

index_type0
{
6SpatialTransformer/_transform/_meshgrid/LinSpace/startConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
z
5SpatialTransformer/_transform/_meshgrid/LinSpace/stopConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
v
4SpatialTransformer/_transform/_meshgrid/LinSpace/numConst*
value	B :*
dtype0*
_output_shapes
: 
?
0SpatialTransformer/_transform/_meshgrid/LinSpaceLinSpace6SpatialTransformer/_transform/_meshgrid/LinSpace/start5SpatialTransformer/_transform/_meshgrid/LinSpace/stop4SpatialTransformer/_transform/_meshgrid/LinSpace/num*
_output_shapes
:*

Tidx0*
T0
x
6SpatialTransformer/_transform/_meshgrid/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B :
?
2SpatialTransformer/_transform/_meshgrid/ExpandDims
ExpandDims0SpatialTransformer/_transform/_meshgrid/LinSpace6SpatialTransformer/_transform/_meshgrid/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:
?
6SpatialTransformer/_transform/_meshgrid/transpose/permConst*
_output_shapes
:*
valueB"       *
dtype0
?
1SpatialTransformer/_transform/_meshgrid/transpose	Transpose2SpatialTransformer/_transform/_meshgrid/ExpandDims6SpatialTransformer/_transform/_meshgrid/transpose/perm*
T0*
_output_shapes

:*
Tperm0
?
.SpatialTransformer/_transform/_meshgrid/MatMulMatMul,SpatialTransformer/_transform/_meshgrid/ones1SpatialTransformer/_transform/_meshgrid/transpose*
_output_shapes

:*
transpose_a( *
transpose_b( *
T0
}
8SpatialTransformer/_transform/_meshgrid/LinSpace_1/startConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
|
7SpatialTransformer/_transform/_meshgrid/LinSpace_1/stopConst*
_output_shapes
: *
valueB
 *  ??*
dtype0
x
6SpatialTransformer/_transform/_meshgrid/LinSpace_1/numConst*
value	B :*
dtype0*
_output_shapes
: 
?
2SpatialTransformer/_transform/_meshgrid/LinSpace_1LinSpace8SpatialTransformer/_transform/_meshgrid/LinSpace_1/start7SpatialTransformer/_transform/_meshgrid/LinSpace_1/stop6SpatialTransformer/_transform/_meshgrid/LinSpace_1/num*
T0*
_output_shapes
:*

Tidx0
z
8SpatialTransformer/_transform/_meshgrid/ExpandDims_1/dimConst*
value	B :*
dtype0*
_output_shapes
: 
?
4SpatialTransformer/_transform/_meshgrid/ExpandDims_1
ExpandDims2SpatialTransformer/_transform/_meshgrid/LinSpace_18SpatialTransformer/_transform/_meshgrid/ExpandDims_1/dim*
_output_shapes

:*

Tdim0*
T0
?
/SpatialTransformer/_transform/_meshgrid/stack_1Const*
valueB"      *
dtype0*
_output_shapes
:
y
4SpatialTransformer/_transform/_meshgrid/ones_1/ConstConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
.SpatialTransformer/_transform/_meshgrid/ones_1Fill/SpatialTransformer/_transform/_meshgrid/stack_14SpatialTransformer/_transform/_meshgrid/ones_1/Const*

index_type0*
_output_shapes

:*
T0
?
0SpatialTransformer/_transform/_meshgrid/MatMul_1MatMul4SpatialTransformer/_transform/_meshgrid/ExpandDims_1.SpatialTransformer/_transform/_meshgrid/ones_1*
T0*
_output_shapes

:*
transpose_a( *
transpose_b( 
?
5SpatialTransformer/_transform/_meshgrid/Reshape/shapeConst*
valueB"   ????*
dtype0*
_output_shapes
:
?
/SpatialTransformer/_transform/_meshgrid/ReshapeReshape.SpatialTransformer/_transform/_meshgrid/MatMul5SpatialTransformer/_transform/_meshgrid/Reshape/shape*
T0*
Tshape0*
_output_shapes
:	?
?
7SpatialTransformer/_transform/_meshgrid/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"   ????
?
1SpatialTransformer/_transform/_meshgrid/Reshape_1Reshape0SpatialTransformer/_transform/_meshgrid/MatMul_17SpatialTransformer/_transform/_meshgrid/Reshape_1/shape*
_output_shapes
:	?*
T0*
Tshape0
?
7SpatialTransformer/_transform/_meshgrid/ones_like/ShapeConst*
_output_shapes
:*
valueB"     *
dtype0
|
7SpatialTransformer/_transform/_meshgrid/ones_like/ConstConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
1SpatialTransformer/_transform/_meshgrid/ones_likeFill7SpatialTransformer/_transform/_meshgrid/ones_like/Shape7SpatialTransformer/_transform/_meshgrid/ones_like/Const*
T0*

index_type0*
_output_shapes
:	?
u
3SpatialTransformer/_transform/_meshgrid/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
.SpatialTransformer/_transform/_meshgrid/concatConcatV2/SpatialTransformer/_transform/_meshgrid/Reshape1SpatialTransformer/_transform/_meshgrid/Reshape_11SpatialTransformer/_transform/_meshgrid/ones_like3SpatialTransformer/_transform/_meshgrid/concat/axis*
N*
_output_shapes
:	?*

Tidx0*
T0
n
,SpatialTransformer/_transform/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
?
(SpatialTransformer/_transform/ExpandDims
ExpandDims.SpatialTransformer/_transform/_meshgrid/concat,SpatialTransformer/_transform/ExpandDims/dim*

Tdim0*
T0*#
_output_shapes
:?
?
-SpatialTransformer/_transform/Reshape_1/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
'SpatialTransformer/_transform/Reshape_1Reshape(SpatialTransformer/_transform/ExpandDims-SpatialTransformer/_transform/Reshape_1/shape*
Tshape0*
_output_shapes	
:?*
T0
?
#SpatialTransformer/_transform/stackPack+SpatialTransformer/_transform/strided_slice*

axis *
N*
_output_shapes
:*
T0
?
"SpatialTransformer/_transform/TileTile'SpatialTransformer/_transform/Reshape_1#SpatialTransformer/_transform/stack*

Tmultiples0*
T0*#
_output_shapes
:?????????
i
'SpatialTransformer/_transform/stack_1/1Const*
value	B :*
dtype0*
_output_shapes
: 
r
'SpatialTransformer/_transform/stack_1/2Const*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
%SpatialTransformer/_transform/stack_1Pack+SpatialTransformer/_transform/strided_slice'SpatialTransformer/_transform/stack_1/1'SpatialTransformer/_transform/stack_1/2*
_output_shapes
:*
T0*

axis *
N
?
'SpatialTransformer/_transform/Reshape_2Reshape"SpatialTransformer/_transform/Tile%SpatialTransformer/_transform/stack_1*
T0*
Tshape0*4
_output_shapes"
 :??????????????????
?
$SpatialTransformer/_transform/MatMulBatchMatMulV2%SpatialTransformer/_transform/Reshape'SpatialTransformer/_transform/Reshape_2*4
_output_shapes"
 :??????????????????*
adj_x( *
adj_y( *
T0
~
)SpatialTransformer/_transform/Slice/beginConst*!
valueB"            *
dtype0*
_output_shapes
:
}
(SpatialTransformer/_transform/Slice/sizeConst*!
valueB"????   ????*
dtype0*
_output_shapes
:
?
#SpatialTransformer/_transform/SliceSlice$SpatialTransformer/_transform/MatMul)SpatialTransformer/_transform/Slice/begin(SpatialTransformer/_transform/Slice/size*
Index0*
T0*4
_output_shapes"
 :??????????????????
?
+SpatialTransformer/_transform/Slice_1/beginConst*!
valueB"           *
dtype0*
_output_shapes
:

*SpatialTransformer/_transform/Slice_1/sizeConst*!
valueB"????   ????*
dtype0*
_output_shapes
:
?
%SpatialTransformer/_transform/Slice_1Slice$SpatialTransformer/_transform/MatMul+SpatialTransformer/_transform/Slice_1/begin*SpatialTransformer/_transform/Slice_1/size*
Index0*
T0*4
_output_shapes"
 :??????????????????
?
-SpatialTransformer/_transform/Reshape_3/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
'SpatialTransformer/_transform/Reshape_3Reshape#SpatialTransformer/_transform/Slice-SpatialTransformer/_transform/Reshape_3/shape*#
_output_shapes
:?????????*
T0*
Tshape0
?
-SpatialTransformer/_transform/Reshape_4/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
?
'SpatialTransformer/_transform/Reshape_4Reshape%SpatialTransformer/_transform/Slice_1-SpatialTransformer/_transform/Reshape_4/shape*
T0*
Tshape0*#
_output_shapes
:?????????
w
0SpatialTransformer/_transform/_interpolate/ShapeShapeReshape*
T0*
out_type0*
_output_shapes
:
?
>SpatialTransformer/_transform/_interpolate/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
?
@SpatialTransformer/_transform/_interpolate/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
@SpatialTransformer/_transform/_interpolate/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
8SpatialTransformer/_transform/_interpolate/strided_sliceStridedSlice0SpatialTransformer/_transform/_interpolate/Shape>SpatialTransformer/_transform/_interpolate/strided_slice/stack@SpatialTransformer/_transform/_interpolate/strided_slice/stack_1@SpatialTransformer/_transform/_interpolate/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
y
2SpatialTransformer/_transform/_interpolate/Shape_1ShapeReshape*
T0*
out_type0*
_output_shapes
:
?
@SpatialTransformer/_transform/_interpolate/strided_slice_1/stackConst*
_output_shapes
:*
valueB:*
dtype0
?
BSpatialTransformer/_transform/_interpolate/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
BSpatialTransformer/_transform/_interpolate/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
:SpatialTransformer/_transform/_interpolate/strided_slice_1StridedSlice2SpatialTransformer/_transform/_interpolate/Shape_1@SpatialTransformer/_transform/_interpolate/strided_slice_1/stackBSpatialTransformer/_transform/_interpolate/strided_slice_1/stack_1BSpatialTransformer/_transform/_interpolate/strided_slice_1/stack_2*
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask 
y
2SpatialTransformer/_transform/_interpolate/Shape_2ShapeReshape*
T0*
out_type0*
_output_shapes
:
?
@SpatialTransformer/_transform/_interpolate/strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB:
?
BSpatialTransformer/_transform/_interpolate/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
BSpatialTransformer/_transform/_interpolate/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
:SpatialTransformer/_transform/_interpolate/strided_slice_2StridedSlice2SpatialTransformer/_transform/_interpolate/Shape_2@SpatialTransformer/_transform/_interpolate/strided_slice_2/stackBSpatialTransformer/_transform/_interpolate/strided_slice_2/stack_1BSpatialTransformer/_transform/_interpolate/strided_slice_2/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
y
2SpatialTransformer/_transform/_interpolate/Shape_3ShapeReshape*
T0*
out_type0*
_output_shapes
:
?
@SpatialTransformer/_transform/_interpolate/strided_slice_3/stackConst*
valueB:*
dtype0*
_output_shapes
:
?
BSpatialTransformer/_transform/_interpolate/strided_slice_3/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
?
BSpatialTransformer/_transform/_interpolate/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
:SpatialTransformer/_transform/_interpolate/strided_slice_3StridedSlice2SpatialTransformer/_transform/_interpolate/Shape_3@SpatialTransformer/_transform/_interpolate/strided_slice_3/stackBSpatialTransformer/_transform/_interpolate/strided_slice_3/stack_1BSpatialTransformer/_transform/_interpolate/strided_slice_3/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
?
/SpatialTransformer/_transform/_interpolate/CastCast:SpatialTransformer/_transform/_interpolate/strided_slice_1*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
?
1SpatialTransformer/_transform/_interpolate/Cast_1Cast:SpatialTransformer/_transform/_interpolate/strided_slice_2*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
r
0SpatialTransformer/_transform/_interpolate/zerosConst*
_output_shapes
: *
value	B : *
dtype0
y
2SpatialTransformer/_transform/_interpolate/Shape_4ShapeReshape*
T0*
out_type0*
_output_shapes
:
?
@SpatialTransformer/_transform/_interpolate/strided_slice_4/stackConst*
valueB:*
dtype0*
_output_shapes
:
?
BSpatialTransformer/_transform/_interpolate/strided_slice_4/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
BSpatialTransformer/_transform/_interpolate/strided_slice_4/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
:SpatialTransformer/_transform/_interpolate/strided_slice_4StridedSlice2SpatialTransformer/_transform/_interpolate/Shape_4@SpatialTransformer/_transform/_interpolate/strided_slice_4/stackBSpatialTransformer/_transform/_interpolate/strided_slice_4/stack_1BSpatialTransformer/_transform/_interpolate/strided_slice_4/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
r
0SpatialTransformer/_transform/_interpolate/sub/yConst*
_output_shapes
: *
value	B :*
dtype0
?
.SpatialTransformer/_transform/_interpolate/subSub:SpatialTransformer/_transform/_interpolate/strided_slice_40SpatialTransformer/_transform/_interpolate/sub/y*
T0*
_output_shapes
: 
y
2SpatialTransformer/_transform/_interpolate/Shape_5ShapeReshape*
T0*
out_type0*
_output_shapes
:
?
@SpatialTransformer/_transform/_interpolate/strided_slice_5/stackConst*
dtype0*
_output_shapes
:*
valueB:
?
BSpatialTransformer/_transform/_interpolate/strided_slice_5/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
?
BSpatialTransformer/_transform/_interpolate/strided_slice_5/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
:SpatialTransformer/_transform/_interpolate/strided_slice_5StridedSlice2SpatialTransformer/_transform/_interpolate/Shape_5@SpatialTransformer/_transform/_interpolate/strided_slice_5/stackBSpatialTransformer/_transform/_interpolate/strided_slice_5/stack_1BSpatialTransformer/_transform/_interpolate/strided_slice_5/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
t
2SpatialTransformer/_transform/_interpolate/sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
?
0SpatialTransformer/_transform/_interpolate/sub_1Sub:SpatialTransformer/_transform/_interpolate/strided_slice_52SpatialTransformer/_transform/_interpolate/sub_1/y*
_output_shapes
: *
T0
u
0SpatialTransformer/_transform/_interpolate/add/yConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
.SpatialTransformer/_transform/_interpolate/addAdd'SpatialTransformer/_transform/Reshape_30SpatialTransformer/_transform/_interpolate/add/y*#
_output_shapes
:?????????*
T0
?
.SpatialTransformer/_transform/_interpolate/mulMul.SpatialTransformer/_transform/_interpolate/add1SpatialTransformer/_transform/_interpolate/Cast_1*
T0*#
_output_shapes
:?????????
y
4SpatialTransformer/_transform/_interpolate/truediv/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
?
2SpatialTransformer/_transform/_interpolate/truedivRealDiv.SpatialTransformer/_transform/_interpolate/mul4SpatialTransformer/_transform/_interpolate/truediv/y*
T0*#
_output_shapes
:?????????
w
2SpatialTransformer/_transform/_interpolate/add_1/yConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
0SpatialTransformer/_transform/_interpolate/add_1Add'SpatialTransformer/_transform/Reshape_42SpatialTransformer/_transform/_interpolate/add_1/y*
T0*#
_output_shapes
:?????????
?
0SpatialTransformer/_transform/_interpolate/mul_1Mul0SpatialTransformer/_transform/_interpolate/add_1/SpatialTransformer/_transform/_interpolate/Cast*#
_output_shapes
:?????????*
T0
{
6SpatialTransformer/_transform/_interpolate/truediv_1/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
?
4SpatialTransformer/_transform/_interpolate/truediv_1RealDiv0SpatialTransformer/_transform/_interpolate/mul_16SpatialTransformer/_transform/_interpolate/truediv_1/y*#
_output_shapes
:?????????*
T0
?
0SpatialTransformer/_transform/_interpolate/FloorFloor2SpatialTransformer/_transform/_interpolate/truediv*#
_output_shapes
:?????????*
T0
?
1SpatialTransformer/_transform/_interpolate/Cast_2Cast0SpatialTransformer/_transform/_interpolate/Floor*

SrcT0*
Truncate( *#
_output_shapes
:?????????*

DstT0
t
2SpatialTransformer/_transform/_interpolate/add_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
?
0SpatialTransformer/_transform/_interpolate/add_2Add1SpatialTransformer/_transform/_interpolate/Cast_22SpatialTransformer/_transform/_interpolate/add_2/y*#
_output_shapes
:?????????*
T0
?
2SpatialTransformer/_transform/_interpolate/Floor_1Floor4SpatialTransformer/_transform/_interpolate/truediv_1*#
_output_shapes
:?????????*
T0
?
1SpatialTransformer/_transform/_interpolate/Cast_3Cast2SpatialTransformer/_transform/_interpolate/Floor_1*

SrcT0*
Truncate( *#
_output_shapes
:?????????*

DstT0
t
2SpatialTransformer/_transform/_interpolate/add_3/yConst*
value	B :*
dtype0*
_output_shapes
: 
?
0SpatialTransformer/_transform/_interpolate/add_3Add1SpatialTransformer/_transform/_interpolate/Cast_32SpatialTransformer/_transform/_interpolate/add_3/y*
T0*#
_output_shapes
:?????????
?
@SpatialTransformer/_transform/_interpolate/clip_by_value/MinimumMinimum1SpatialTransformer/_transform/_interpolate/Cast_20SpatialTransformer/_transform/_interpolate/sub_1*#
_output_shapes
:?????????*
T0
?
8SpatialTransformer/_transform/_interpolate/clip_by_valueMaximum@SpatialTransformer/_transform/_interpolate/clip_by_value/Minimum0SpatialTransformer/_transform/_interpolate/zeros*
T0*#
_output_shapes
:?????????
?
BSpatialTransformer/_transform/_interpolate/clip_by_value_1/MinimumMinimum0SpatialTransformer/_transform/_interpolate/add_20SpatialTransformer/_transform/_interpolate/sub_1*
T0*#
_output_shapes
:?????????
?
:SpatialTransformer/_transform/_interpolate/clip_by_value_1MaximumBSpatialTransformer/_transform/_interpolate/clip_by_value_1/Minimum0SpatialTransformer/_transform/_interpolate/zeros*#
_output_shapes
:?????????*
T0
?
BSpatialTransformer/_transform/_interpolate/clip_by_value_2/MinimumMinimum1SpatialTransformer/_transform/_interpolate/Cast_3.SpatialTransformer/_transform/_interpolate/sub*#
_output_shapes
:?????????*
T0
?
:SpatialTransformer/_transform/_interpolate/clip_by_value_2MaximumBSpatialTransformer/_transform/_interpolate/clip_by_value_2/Minimum0SpatialTransformer/_transform/_interpolate/zeros*#
_output_shapes
:?????????*
T0
?
BSpatialTransformer/_transform/_interpolate/clip_by_value_3/MinimumMinimum0SpatialTransformer/_transform/_interpolate/add_3.SpatialTransformer/_transform/_interpolate/sub*
T0*#
_output_shapes
:?????????
?
:SpatialTransformer/_transform/_interpolate/clip_by_value_3MaximumBSpatialTransformer/_transform/_interpolate/clip_by_value_3/Minimum0SpatialTransformer/_transform/_interpolate/zeros*#
_output_shapes
:?????????*
T0
?
0SpatialTransformer/_transform/_interpolate/mul_2Mul:SpatialTransformer/_transform/_interpolate/strided_slice_2:SpatialTransformer/_transform/_interpolate/strided_slice_1*
T0*
_output_shapes
: 
x
6SpatialTransformer/_transform/_interpolate/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
x
6SpatialTransformer/_transform/_interpolate/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
?
0SpatialTransformer/_transform/_interpolate/rangeRange6SpatialTransformer/_transform/_interpolate/range/start8SpatialTransformer/_transform/_interpolate/strided_slice6SpatialTransformer/_transform/_interpolate/range/delta*#
_output_shapes
:?????????*

Tidx0
?
0SpatialTransformer/_transform/_interpolate/mul_3Mul0SpatialTransformer/_transform/_interpolate/range0SpatialTransformer/_transform/_interpolate/mul_2*
T0*#
_output_shapes
:?????????
?
8SpatialTransformer/_transform/_interpolate/_repeat/stackConst*
valueB:?*
dtype0*
_output_shapes
:
?
=SpatialTransformer/_transform/_interpolate/_repeat/ones/ConstConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
7SpatialTransformer/_transform/_interpolate/_repeat/onesFill8SpatialTransformer/_transform/_interpolate/_repeat/stack=SpatialTransformer/_transform/_interpolate/_repeat/ones/Const*
T0*

index_type0*
_output_shapes	
:?
?
ASpatialTransformer/_transform/_interpolate/_repeat/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
?
=SpatialTransformer/_transform/_interpolate/_repeat/ExpandDims
ExpandDims7SpatialTransformer/_transform/_interpolate/_repeat/onesASpatialTransformer/_transform/_interpolate/_repeat/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:	?
?
ASpatialTransformer/_transform/_interpolate/_repeat/transpose/permConst*
dtype0*
_output_shapes
:*
valueB"       
?
<SpatialTransformer/_transform/_interpolate/_repeat/transpose	Transpose=SpatialTransformer/_transform/_interpolate/_repeat/ExpandDimsASpatialTransformer/_transform/_interpolate/_repeat/transpose/perm*
_output_shapes
:	?*
Tperm0*
T0
?
7SpatialTransformer/_transform/_interpolate/_repeat/CastCast<SpatialTransformer/_transform/_interpolate/_repeat/transpose*
_output_shapes
:	?*

DstT0*

SrcT0*
Truncate( 
?
@SpatialTransformer/_transform/_interpolate/_repeat/Reshape/shapeConst*
valueB"????   *
dtype0*
_output_shapes
:
?
:SpatialTransformer/_transform/_interpolate/_repeat/ReshapeReshape0SpatialTransformer/_transform/_interpolate/mul_3@SpatialTransformer/_transform/_interpolate/_repeat/Reshape/shape*'
_output_shapes
:?????????*
T0*
Tshape0
?
9SpatialTransformer/_transform/_interpolate/_repeat/MatMulMatMul:SpatialTransformer/_transform/_interpolate/_repeat/Reshape7SpatialTransformer/_transform/_interpolate/_repeat/Cast*
T0*(
_output_shapes
:??????????*
transpose_a( *
transpose_b( 
?
BSpatialTransformer/_transform/_interpolate/_repeat/Reshape_1/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
<SpatialTransformer/_transform/_interpolate/_repeat/Reshape_1Reshape9SpatialTransformer/_transform/_interpolate/_repeat/MatMulBSpatialTransformer/_transform/_interpolate/_repeat/Reshape_1/shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
0SpatialTransformer/_transform/_interpolate/mul_4Mul:SpatialTransformer/_transform/_interpolate/clip_by_value_2:SpatialTransformer/_transform/_interpolate/strided_slice_2*#
_output_shapes
:?????????*
T0
?
0SpatialTransformer/_transform/_interpolate/add_4Add<SpatialTransformer/_transform/_interpolate/_repeat/Reshape_10SpatialTransformer/_transform/_interpolate/mul_4*
T0*#
_output_shapes
:?????????
?
0SpatialTransformer/_transform/_interpolate/mul_5Mul:SpatialTransformer/_transform/_interpolate/clip_by_value_3:SpatialTransformer/_transform/_interpolate/strided_slice_2*
T0*#
_output_shapes
:?????????
?
0SpatialTransformer/_transform/_interpolate/add_5Add<SpatialTransformer/_transform/_interpolate/_repeat/Reshape_10SpatialTransformer/_transform/_interpolate/mul_5*
T0*#
_output_shapes
:?????????
?
0SpatialTransformer/_transform/_interpolate/add_6Add0SpatialTransformer/_transform/_interpolate/add_48SpatialTransformer/_transform/_interpolate/clip_by_value*
T0*#
_output_shapes
:?????????
?
0SpatialTransformer/_transform/_interpolate/add_7Add0SpatialTransformer/_transform/_interpolate/add_58SpatialTransformer/_transform/_interpolate/clip_by_value*
T0*#
_output_shapes
:?????????
?
0SpatialTransformer/_transform/_interpolate/add_8Add0SpatialTransformer/_transform/_interpolate/add_4:SpatialTransformer/_transform/_interpolate/clip_by_value_1*
T0*#
_output_shapes
:?????????
?
0SpatialTransformer/_transform/_interpolate/add_9Add0SpatialTransformer/_transform/_interpolate/add_5:SpatialTransformer/_transform/_interpolate/clip_by_value_1*
T0*#
_output_shapes
:?????????
}
2SpatialTransformer/_transform/_interpolate/stack/0Const*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
0SpatialTransformer/_transform/_interpolate/stackPack2SpatialTransformer/_transform/_interpolate/stack/0:SpatialTransformer/_transform/_interpolate/strided_slice_3*
T0*

axis *
N*
_output_shapes
:
?
2SpatialTransformer/_transform/_interpolate/ReshapeReshapeReshape0SpatialTransformer/_transform/_interpolate/stack*
T0*
Tshape0*0
_output_shapes
:??????????????????
z
8SpatialTransformer/_transform/_interpolate/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
3SpatialTransformer/_transform/_interpolate/GatherV2GatherV22SpatialTransformer/_transform/_interpolate/Reshape0SpatialTransformer/_transform/_interpolate/add_68SpatialTransformer/_transform/_interpolate/GatherV2/axis*

batch_dims *
Tindices0*
Tparams0*0
_output_shapes
:??????????????????*
Taxis0
|
:SpatialTransformer/_transform/_interpolate/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
5SpatialTransformer/_transform/_interpolate/GatherV2_1GatherV22SpatialTransformer/_transform/_interpolate/Reshape0SpatialTransformer/_transform/_interpolate/add_7:SpatialTransformer/_transform/_interpolate/GatherV2_1/axis*0
_output_shapes
:??????????????????*
Taxis0*

batch_dims *
Tindices0*
Tparams0
|
:SpatialTransformer/_transform/_interpolate/GatherV2_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
?
5SpatialTransformer/_transform/_interpolate/GatherV2_2GatherV22SpatialTransformer/_transform/_interpolate/Reshape0SpatialTransformer/_transform/_interpolate/add_8:SpatialTransformer/_transform/_interpolate/GatherV2_2/axis*
Tindices0*
Tparams0*0
_output_shapes
:??????????????????*
Taxis0*

batch_dims 
|
:SpatialTransformer/_transform/_interpolate/GatherV2_3/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
5SpatialTransformer/_transform/_interpolate/GatherV2_3GatherV22SpatialTransformer/_transform/_interpolate/Reshape0SpatialTransformer/_transform/_interpolate/add_9:SpatialTransformer/_transform/_interpolate/GatherV2_3/axis*0
_output_shapes
:??????????????????*
Taxis0*

batch_dims *
Tindices0*
Tparams0
?
1SpatialTransformer/_transform/_interpolate/Cast_4Cast8SpatialTransformer/_transform/_interpolate/clip_by_value*

SrcT0*
Truncate( *#
_output_shapes
:?????????*

DstT0
?
1SpatialTransformer/_transform/_interpolate/Cast_5Cast:SpatialTransformer/_transform/_interpolate/clip_by_value_1*

SrcT0*
Truncate( *#
_output_shapes
:?????????*

DstT0
?
1SpatialTransformer/_transform/_interpolate/Cast_6Cast:SpatialTransformer/_transform/_interpolate/clip_by_value_2*#
_output_shapes
:?????????*

DstT0*

SrcT0*
Truncate( 
?
1SpatialTransformer/_transform/_interpolate/Cast_7Cast:SpatialTransformer/_transform/_interpolate/clip_by_value_3*

SrcT0*
Truncate( *#
_output_shapes
:?????????*

DstT0
?
0SpatialTransformer/_transform/_interpolate/sub_2Sub1SpatialTransformer/_transform/_interpolate/Cast_52SpatialTransformer/_transform/_interpolate/truediv*
T0*#
_output_shapes
:?????????
?
0SpatialTransformer/_transform/_interpolate/sub_3Sub1SpatialTransformer/_transform/_interpolate/Cast_74SpatialTransformer/_transform/_interpolate/truediv_1*
T0*#
_output_shapes
:?????????
?
0SpatialTransformer/_transform/_interpolate/mul_6Mul0SpatialTransformer/_transform/_interpolate/sub_20SpatialTransformer/_transform/_interpolate/sub_3*#
_output_shapes
:?????????*
T0
{
9SpatialTransformer/_transform/_interpolate/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
?
5SpatialTransformer/_transform/_interpolate/ExpandDims
ExpandDims0SpatialTransformer/_transform/_interpolate/mul_69SpatialTransformer/_transform/_interpolate/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:?????????
?
0SpatialTransformer/_transform/_interpolate/sub_4Sub1SpatialTransformer/_transform/_interpolate/Cast_52SpatialTransformer/_transform/_interpolate/truediv*
T0*#
_output_shapes
:?????????
?
0SpatialTransformer/_transform/_interpolate/sub_5Sub4SpatialTransformer/_transform/_interpolate/truediv_11SpatialTransformer/_transform/_interpolate/Cast_6*
T0*#
_output_shapes
:?????????
?
0SpatialTransformer/_transform/_interpolate/mul_7Mul0SpatialTransformer/_transform/_interpolate/sub_40SpatialTransformer/_transform/_interpolate/sub_5*
T0*#
_output_shapes
:?????????
}
;SpatialTransformer/_transform/_interpolate/ExpandDims_1/dimConst*
value	B :*
dtype0*
_output_shapes
: 
?
7SpatialTransformer/_transform/_interpolate/ExpandDims_1
ExpandDims0SpatialTransformer/_transform/_interpolate/mul_7;SpatialTransformer/_transform/_interpolate/ExpandDims_1/dim*

Tdim0*
T0*'
_output_shapes
:?????????
?
0SpatialTransformer/_transform/_interpolate/sub_6Sub2SpatialTransformer/_transform/_interpolate/truediv1SpatialTransformer/_transform/_interpolate/Cast_4*
T0*#
_output_shapes
:?????????
?
0SpatialTransformer/_transform/_interpolate/sub_7Sub1SpatialTransformer/_transform/_interpolate/Cast_74SpatialTransformer/_transform/_interpolate/truediv_1*
T0*#
_output_shapes
:?????????
?
0SpatialTransformer/_transform/_interpolate/mul_8Mul0SpatialTransformer/_transform/_interpolate/sub_60SpatialTransformer/_transform/_interpolate/sub_7*#
_output_shapes
:?????????*
T0
}
;SpatialTransformer/_transform/_interpolate/ExpandDims_2/dimConst*
value	B :*
dtype0*
_output_shapes
: 
?
7SpatialTransformer/_transform/_interpolate/ExpandDims_2
ExpandDims0SpatialTransformer/_transform/_interpolate/mul_8;SpatialTransformer/_transform/_interpolate/ExpandDims_2/dim*'
_output_shapes
:?????????*

Tdim0*
T0
?
0SpatialTransformer/_transform/_interpolate/sub_8Sub2SpatialTransformer/_transform/_interpolate/truediv1SpatialTransformer/_transform/_interpolate/Cast_4*
T0*#
_output_shapes
:?????????
?
0SpatialTransformer/_transform/_interpolate/sub_9Sub4SpatialTransformer/_transform/_interpolate/truediv_11SpatialTransformer/_transform/_interpolate/Cast_6*
T0*#
_output_shapes
:?????????
?
0SpatialTransformer/_transform/_interpolate/mul_9Mul0SpatialTransformer/_transform/_interpolate/sub_80SpatialTransformer/_transform/_interpolate/sub_9*
T0*#
_output_shapes
:?????????
}
;SpatialTransformer/_transform/_interpolate/ExpandDims_3/dimConst*
value	B :*
dtype0*
_output_shapes
: 
?
7SpatialTransformer/_transform/_interpolate/ExpandDims_3
ExpandDims0SpatialTransformer/_transform/_interpolate/mul_9;SpatialTransformer/_transform/_interpolate/ExpandDims_3/dim*

Tdim0*
T0*'
_output_shapes
:?????????
?
1SpatialTransformer/_transform/_interpolate/mul_10Mul5SpatialTransformer/_transform/_interpolate/ExpandDims3SpatialTransformer/_transform/_interpolate/GatherV2*0
_output_shapes
:??????????????????*
T0
?
1SpatialTransformer/_transform/_interpolate/mul_11Mul7SpatialTransformer/_transform/_interpolate/ExpandDims_15SpatialTransformer/_transform/_interpolate/GatherV2_1*0
_output_shapes
:??????????????????*
T0
?
1SpatialTransformer/_transform/_interpolate/mul_12Mul7SpatialTransformer/_transform/_interpolate/ExpandDims_25SpatialTransformer/_transform/_interpolate/GatherV2_2*
T0*0
_output_shapes
:??????????????????
?
1SpatialTransformer/_transform/_interpolate/mul_13Mul7SpatialTransformer/_transform/_interpolate/ExpandDims_35SpatialTransformer/_transform/_interpolate/GatherV2_3*
T0*0
_output_shapes
:??????????????????
?
/SpatialTransformer/_transform/_interpolate/AddNAddN1SpatialTransformer/_transform/_interpolate/mul_101SpatialTransformer/_transform/_interpolate/mul_111SpatialTransformer/_transform/_interpolate/mul_121SpatialTransformer/_transform/_interpolate/mul_13*
T0*
N*0
_output_shapes
:??????????????????
i
'SpatialTransformer/_transform/stack_2/1Const*
value	B :*
dtype0*
_output_shapes
: 
i
'SpatialTransformer/_transform/stack_2/2Const*
value	B :*
dtype0*
_output_shapes
: 
?
%SpatialTransformer/_transform/stack_2Pack+SpatialTransformer/_transform/strided_slice'SpatialTransformer/_transform/stack_2/1'SpatialTransformer/_transform/stack_2/2-SpatialTransformer/_transform/strided_slice_3*

axis *
N*
_output_shapes
:*
T0
?
'SpatialTransformer/_transform/Reshape_5Reshape/SpatialTransformer/_transform/_interpolate/AddN%SpatialTransformer/_transform/stack_2*8
_output_shapes&
$:"??????????????????*
T0*
Tshape0
q
truncated_normal_2/shapeConst*%
valueB"             *
dtype0*
_output_shapes
:
\
truncated_normal_2/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_2/stddevConst*
valueB
 *???=*
dtype0*
_output_shapes
: 
?
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
dtype0*&
_output_shapes
: *
seed2 *

seed *
T0
?
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*&
_output_shapes
: *
T0
{
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0*&
_output_shapes
: 
?

Variable_3
VariableV2*
dtype0*&
_output_shapes
: *
	container *
shape: *
shared_name 
?
Variable_3/AssignAssign
Variable_3truncated_normal_2*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*&
_output_shapes
: 
w
Variable_3/readIdentity
Variable_3*
T0*
_class
loc:@Variable_3*&
_output_shapes
: 
T
Const_1Const*
dtype0*
_output_shapes
: *
valueB *???=
v

Variable_4
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
?
Variable_4/AssignAssign
Variable_4Const_1*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(
k
Variable_4/readIdentity
Variable_4*
T0*
_class
loc:@Variable_4*
_output_shapes
: 
?
Conv2DConv2D'SpatialTransformer/_transform/Reshape_5Variable_3/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:????????? 
_
add_2AddConv2DVariable_4/read*
T0*/
_output_shapes
:????????? 
M
ReluReluadd_2*/
_output_shapes
:????????? *
T0
?
MaxPoolMaxPoolRelu*
ksize
*
paddingSAME*/
_output_shapes
:????????? *
T0*
data_formatNHWC*
strides

q
truncated_normal_3/shapeConst*%
valueB"          @   *
dtype0*
_output_shapes
:
\
truncated_normal_3/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
^
truncated_normal_3/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *???=
?
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
dtype0*&
_output_shapes
: @*
seed2 *

seed *
T0
?
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*&
_output_shapes
: @*
T0
{
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
T0*&
_output_shapes
: @
?

Variable_5
VariableV2*
shape: @*
shared_name *
dtype0*&
_output_shapes
: @*
	container 
?
Variable_5/AssignAssign
Variable_5truncated_normal_3*
_class
loc:@Variable_5*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0
w
Variable_5/readIdentity
Variable_5*
T0*
_class
loc:@Variable_5*&
_output_shapes
: @
T
Const_2Const*
dtype0*
_output_shapes
:@*
valueB@*???=
v

Variable_6
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 
?
Variable_6/AssignAssign
Variable_6Const_2*
T0*
_class
loc:@Variable_6*
validate_shape(*
_output_shapes
:@*
use_locking(
k
Variable_6/readIdentity
Variable_6*
T0*
_class
loc:@Variable_6*
_output_shapes
:@
?
Conv2D_1Conv2DMaxPoolVariable_5/read*
paddingSAME*/
_output_shapes
:?????????@*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 
a
add_3AddConv2D_1Variable_6/read*
T0*/
_output_shapes
:?????????@
O
Relu_1Reluadd_3*
T0*/
_output_shapes
:?????????@
?
	MaxPool_1MaxPoolRelu_1*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*/
_output_shapes
:?????????@
i
truncated_normal_4/shapeConst*
valueB"@     *
dtype0*
_output_shapes
:
\
truncated_normal_4/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_4/stddevConst*
valueB
 *???=*
dtype0*
_output_shapes
: 
?
"truncated_normal_4/TruncatedNormalTruncatedNormaltruncated_normal_4/shape*
T0*
dtype0*
_output_shapes
:	?*
seed2 *

seed 
?
truncated_normal_4/mulMul"truncated_normal_4/TruncatedNormaltruncated_normal_4/stddev*
T0*
_output_shapes
:	?
t
truncated_normal_4Addtruncated_normal_4/multruncated_normal_4/mean*
T0*
_output_shapes
:	?
?

Variable_7
VariableV2*
dtype0*
_output_shapes
:	?*
	container *
shape:	?*
shared_name 
?
Variable_7/AssignAssign
Variable_7truncated_normal_4*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes
:	?*
use_locking(
p
Variable_7/readIdentity
Variable_7*
_output_shapes
:	?*
T0*
_class
loc:@Variable_7
T
Const_3Const*
valueB*???=*
dtype0*
_output_shapes
:
v

Variable_8
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
?
Variable_8/AssignAssign
Variable_8Const_3*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@Variable_8
k
Variable_8/readIdentity
Variable_8*
_output_shapes
:*
T0*
_class
loc:@Variable_8
`
Reshape_1/shapeConst*
valueB"????@  *
dtype0*
_output_shapes
:
q
	Reshape_1Reshape	MaxPool_1Reshape_1/shape*(
_output_shapes
:??????????*
T0*
Tshape0
?
MatMul_2MatMul	Reshape_1Variable_7/read*
transpose_b( *
T0*'
_output_shapes
:?????????*
transpose_a( 
Y
add_4AddMatMul_2Variable_8/read*
T0*'
_output_shapes
:?????????
G
Relu_2Reluadd_4*'
_output_shapes
:?????????*
T0
L
sub_1/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
C
sub_1Subsub_1/x	keep_prob*
T0*
_output_shapes
:
U
dropout_1/ShapeShapeRelu_2*
T0*
out_type0*
_output_shapes
:
a
dropout_1/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
a
dropout_1/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  ??
?
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape*
T0*
dtype0*'
_output_shapes
:?????????*
seed2 *

seed 
?
dropout_1/random_uniform/subSubdropout_1/random_uniform/maxdropout_1/random_uniform/min*
T0*
_output_shapes
: 
?
dropout_1/random_uniform/mulMul&dropout_1/random_uniform/RandomUniformdropout_1/random_uniform/sub*'
_output_shapes
:?????????*
T0
?
dropout_1/random_uniformAdddropout_1/random_uniform/muldropout_1/random_uniform/min*'
_output_shapes
:?????????*
T0
T
dropout_1/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ??
O
dropout_1/subSubdropout_1/sub/xsub_1*
T0*
_output_shapes
:
X
dropout_1/truediv/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
c
dropout_1/truedivRealDivdropout_1/truediv/xdropout_1/sub*
T0*
_output_shapes
:
j
dropout_1/GreaterEqualGreaterEqualdropout_1/random_uniformsub_1*
_output_shapes
:*
T0
R
dropout_1/mulMulRelu_2dropout_1/truediv*
_output_shapes
:*
T0
p
dropout_1/CastCastdropout_1/GreaterEqual*
_output_shapes
:*

DstT0*

SrcT0
*
Truncate( 
g
dropout_1/mul_1Muldropout_1/muldropout_1/Cast*'
_output_shapes
:?????????*
T0
i
truncated_normal_5/shapeConst*
_output_shapes
:*
valueB"   
   *
dtype0
\
truncated_normal_5/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_5/stddevConst*
valueB
 *???=*
dtype0*
_output_shapes
: 
?
"truncated_normal_5/TruncatedNormalTruncatedNormaltruncated_normal_5/shape*
_output_shapes

:
*
seed2 *

seed *
T0*
dtype0
?
truncated_normal_5/mulMul"truncated_normal_5/TruncatedNormaltruncated_normal_5/stddev*
T0*
_output_shapes

:

s
truncated_normal_5Addtruncated_normal_5/multruncated_normal_5/mean*
T0*
_output_shapes

:

~

Variable_9
VariableV2*
_output_shapes

:
*
	container *
shape
:
*
shared_name *
dtype0
?
Variable_9/AssignAssign
Variable_9truncated_normal_5*
T0*
_class
loc:@Variable_9*
validate_shape(*
_output_shapes

:
*
use_locking(
o
Variable_9/readIdentity
Variable_9*
_output_shapes

:
*
T0*
_class
loc:@Variable_9
T
Const_4Const*
dtype0*
_output_shapes
:
*
valueB
*???=
w
Variable_10
VariableV2*
_output_shapes
:
*
	container *
shape:
*
shared_name *
dtype0
?
Variable_10/AssignAssignVariable_10Const_4*
_class
loc:@Variable_10*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0
n
Variable_10/readIdentityVariable_10*
_output_shapes
:
*
T0*
_class
loc:@Variable_10
?
MatMul_3MatMuldropout_1/mul_1Variable_9/read*
T0*'
_output_shapes
:?????????
*
transpose_a( *
transpose_b( 
[
y_convAddMatMul_3Variable_10/read*
T0*'
_output_shapes
:?????????


9softmax_cross_entropy_with_logits_sg/labels_stop_gradientStopGradienty_*
T0*'
_output_shapes
:?????????

k
)softmax_cross_entropy_with_logits_sg/RankConst*
value	B :*
dtype0*
_output_shapes
: 
p
*softmax_cross_entropy_with_logits_sg/ShapeShapey_conv*
T0*
out_type0*
_output_shapes
:
m
+softmax_cross_entropy_with_logits_sg/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
r
,softmax_cross_entropy_with_logits_sg/Shape_1Shapey_conv*
T0*
out_type0*
_output_shapes
:
l
*softmax_cross_entropy_with_logits_sg/Sub/yConst*
_output_shapes
: *
value	B :*
dtype0
?
(softmax_cross_entropy_with_logits_sg/SubSub+softmax_cross_entropy_with_logits_sg/Rank_1*softmax_cross_entropy_with_logits_sg/Sub/y*
T0*
_output_shapes
: 
?
0softmax_cross_entropy_with_logits_sg/Slice/beginPack(softmax_cross_entropy_with_logits_sg/Sub*
T0*

axis *
N*
_output_shapes
:
y
/softmax_cross_entropy_with_logits_sg/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
?
*softmax_cross_entropy_with_logits_sg/SliceSlice,softmax_cross_entropy_with_logits_sg/Shape_10softmax_cross_entropy_with_logits_sg/Slice/begin/softmax_cross_entropy_with_logits_sg/Slice/size*
_output_shapes
:*
Index0*
T0
?
4softmax_cross_entropy_with_logits_sg/concat/values_0Const*
dtype0*
_output_shapes
:*
valueB:
?????????
r
0softmax_cross_entropy_with_logits_sg/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
+softmax_cross_entropy_with_logits_sg/concatConcatV24softmax_cross_entropy_with_logits_sg/concat/values_0*softmax_cross_entropy_with_logits_sg/Slice0softmax_cross_entropy_with_logits_sg/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
?
,softmax_cross_entropy_with_logits_sg/ReshapeReshapey_conv+softmax_cross_entropy_with_logits_sg/concat*
T0*
Tshape0*0
_output_shapes
:??????????????????
m
+softmax_cross_entropy_with_logits_sg/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
?
,softmax_cross_entropy_with_logits_sg/Shape_2Shape9softmax_cross_entropy_with_logits_sg/labels_stop_gradient*
_output_shapes
:*
T0*
out_type0
n
,softmax_cross_entropy_with_logits_sg/Sub_1/yConst*
dtype0*
_output_shapes
: *
value	B :
?
*softmax_cross_entropy_with_logits_sg/Sub_1Sub+softmax_cross_entropy_with_logits_sg/Rank_2,softmax_cross_entropy_with_logits_sg/Sub_1/y*
T0*
_output_shapes
: 
?
2softmax_cross_entropy_with_logits_sg/Slice_1/beginPack*softmax_cross_entropy_with_logits_sg/Sub_1*
N*
_output_shapes
:*
T0*

axis 
{
1softmax_cross_entropy_with_logits_sg/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
?
,softmax_cross_entropy_with_logits_sg/Slice_1Slice,softmax_cross_entropy_with_logits_sg/Shape_22softmax_cross_entropy_with_logits_sg/Slice_1/begin1softmax_cross_entropy_with_logits_sg/Slice_1/size*
_output_shapes
:*
Index0*
T0
?
6softmax_cross_entropy_with_logits_sg/concat_1/values_0Const*
valueB:
?????????*
dtype0*
_output_shapes
:
t
2softmax_cross_entropy_with_logits_sg/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
-softmax_cross_entropy_with_logits_sg/concat_1ConcatV26softmax_cross_entropy_with_logits_sg/concat_1/values_0,softmax_cross_entropy_with_logits_sg/Slice_12softmax_cross_entropy_with_logits_sg/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
?
.softmax_cross_entropy_with_logits_sg/Reshape_1Reshape9softmax_cross_entropy_with_logits_sg/labels_stop_gradient-softmax_cross_entropy_with_logits_sg/concat_1*
T0*
Tshape0*0
_output_shapes
:??????????????????
?
$softmax_cross_entropy_with_logits_sgSoftmaxCrossEntropyWithLogits,softmax_cross_entropy_with_logits_sg/Reshape.softmax_cross_entropy_with_logits_sg/Reshape_1*?
_output_shapes-
+:?????????:??????????????????*
T0
n
,softmax_cross_entropy_with_logits_sg/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
?
*softmax_cross_entropy_with_logits_sg/Sub_2Sub)softmax_cross_entropy_with_logits_sg/Rank,softmax_cross_entropy_with_logits_sg/Sub_2/y*
T0*
_output_shapes
: 
|
2softmax_cross_entropy_with_logits_sg/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
?
1softmax_cross_entropy_with_logits_sg/Slice_2/sizePack*softmax_cross_entropy_with_logits_sg/Sub_2*
T0*

axis *
N*
_output_shapes
:
?
,softmax_cross_entropy_with_logits_sg/Slice_2Slice*softmax_cross_entropy_with_logits_sg/Shape2softmax_cross_entropy_with_logits_sg/Slice_2/begin1softmax_cross_entropy_with_logits_sg/Slice_2/size*
_output_shapes
:*
Index0*
T0
?
.softmax_cross_entropy_with_logits_sg/Reshape_2Reshape$softmax_cross_entropy_with_logits_sg,softmax_cross_entropy_with_logits_sg/Slice_2*
T0*
Tshape0*#
_output_shapes
:?????????
Q
Const_5Const*
valueB: *
dtype0*
_output_shapes
:
?
MeanMean.softmax_cross_entropy_with_logits_sg/Reshape_2Const_5*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  ??*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
?
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
?
gradients/Mean_grad/ShapeShape.softmax_cross_entropy_with_logits_sg/Reshape_2*
T0*
out_type0*
_output_shapes
:
?
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:?????????
?
gradients/Mean_grad/Shape_1Shape.softmax_cross_entropy_with_logits_sg/Reshape_2*
out_type0*
_output_shapes
:*
T0
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
?
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
e
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
?
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
?
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *
T0
?
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
~
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
?
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:?????????
?
Cgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ShapeShape$softmax_cross_entropy_with_logits_sg*
T0*
out_type0*
_output_shapes
:
?
Egradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeReshapegradients/Mean_grad/truedivCgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
gradients/zeros_like	ZerosLike&softmax_cross_entropy_with_logits_sg:1*
T0*0
_output_shapes
:??????????????????
?
Bgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dimConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
>gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims
ExpandDimsEgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeBgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:?????????
?
7gradients/softmax_cross_entropy_with_logits_sg_grad/mulMul>gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims&softmax_cross_entropy_with_logits_sg:1*
T0*0
_output_shapes
:??????????????????
?
>gradients/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax
LogSoftmax,softmax_cross_entropy_with_logits_sg/Reshape*0
_output_shapes
:??????????????????*
T0
?
7gradients/softmax_cross_entropy_with_logits_sg_grad/NegNeg>gradients/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax*0
_output_shapes
:??????????????????*
T0
?
Dgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dimConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
@gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1
ExpandDimsEgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeDgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dim*'
_output_shapes
:?????????*

Tdim0*
T0
?
9gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1Mul@gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_17gradients/softmax_cross_entropy_with_logits_sg_grad/Neg*
T0*0
_output_shapes
:??????????????????
?
Dgradients/softmax_cross_entropy_with_logits_sg_grad/tuple/group_depsNoOp8^gradients/softmax_cross_entropy_with_logits_sg_grad/mul:^gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1
?
Lgradients/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_with_logits_sg_grad/mulE^gradients/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/softmax_cross_entropy_with_logits_sg_grad/mul*0
_output_shapes
:??????????????????
?
Ngradients/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1E^gradients/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1*0
_output_shapes
:??????????????????
?
Agradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/ShapeShapey_conv*
T0*
out_type0*
_output_shapes
:
?
Cgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/ReshapeReshapeLgradients/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependencyAgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????

c
gradients/y_conv_grad/ShapeShapeMatMul_3*
T0*
out_type0*
_output_shapes
:
g
gradients/y_conv_grad/Shape_1Const*
_output_shapes
:*
valueB:
*
dtype0
?
+gradients/y_conv_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/y_conv_grad/Shapegradients/y_conv_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients/y_conv_grad/SumSumCgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape+gradients/y_conv_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
gradients/y_conv_grad/ReshapeReshapegradients/y_conv_grad/Sumgradients/y_conv_grad/Shape*'
_output_shapes
:?????????
*
T0*
Tshape0
?
gradients/y_conv_grad/Sum_1SumCgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape-gradients/y_conv_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
gradients/y_conv_grad/Reshape_1Reshapegradients/y_conv_grad/Sum_1gradients/y_conv_grad/Shape_1*
_output_shapes
:
*
T0*
Tshape0
p
&gradients/y_conv_grad/tuple/group_depsNoOp^gradients/y_conv_grad/Reshape ^gradients/y_conv_grad/Reshape_1
?
.gradients/y_conv_grad/tuple/control_dependencyIdentitygradients/y_conv_grad/Reshape'^gradients/y_conv_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/y_conv_grad/Reshape*'
_output_shapes
:?????????

?
0gradients/y_conv_grad/tuple/control_dependency_1Identitygradients/y_conv_grad/Reshape_1'^gradients/y_conv_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/y_conv_grad/Reshape_1*
_output_shapes
:

?
gradients/MatMul_3_grad/MatMulMatMul.gradients/y_conv_grad/tuple/control_dependencyVariable_9/read*
T0*'
_output_shapes
:?????????*
transpose_a( *
transpose_b(
?
 gradients/MatMul_3_grad/MatMul_1MatMuldropout_1/mul_1.gradients/y_conv_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
t
(gradients/MatMul_3_grad/tuple/group_depsNoOp^gradients/MatMul_3_grad/MatMul!^gradients/MatMul_3_grad/MatMul_1
?
0gradients/MatMul_3_grad/tuple/control_dependencyIdentitygradients/MatMul_3_grad/MatMul)^gradients/MatMul_3_grad/tuple/group_deps*'
_output_shapes
:?????????*
T0*1
_class'
%#loc:@gradients/MatMul_3_grad/MatMul
?
2gradients/MatMul_3_grad/tuple/control_dependency_1Identity gradients/MatMul_3_grad/MatMul_1)^gradients/MatMul_3_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_3_grad/MatMul_1*
_output_shapes

:

z
$gradients/dropout_1/mul_1_grad/ShapeShapedropout_1/mul*
out_type0*#
_output_shapes
:?????????*
T0
}
&gradients/dropout_1/mul_1_grad/Shape_1Shapedropout_1/Cast*
T0*
out_type0*#
_output_shapes
:?????????
?
4gradients/dropout_1/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/dropout_1/mul_1_grad/Shape&gradients/dropout_1/mul_1_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
"gradients/dropout_1/mul_1_grad/MulMul0gradients/MatMul_3_grad/tuple/control_dependencydropout_1/Cast*
T0*
_output_shapes
:
?
"gradients/dropout_1/mul_1_grad/SumSum"gradients/dropout_1/mul_1_grad/Mul4gradients/dropout_1/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
&gradients/dropout_1/mul_1_grad/ReshapeReshape"gradients/dropout_1/mul_1_grad/Sum$gradients/dropout_1/mul_1_grad/Shape*
T0*
Tshape0*
_output_shapes
:
?
$gradients/dropout_1/mul_1_grad/Mul_1Muldropout_1/mul0gradients/MatMul_3_grad/tuple/control_dependency*
_output_shapes
:*
T0
?
$gradients/dropout_1/mul_1_grad/Sum_1Sum$gradients/dropout_1/mul_1_grad/Mul_16gradients/dropout_1/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
?
(gradients/dropout_1/mul_1_grad/Reshape_1Reshape$gradients/dropout_1/mul_1_grad/Sum_1&gradients/dropout_1/mul_1_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
?
/gradients/dropout_1/mul_1_grad/tuple/group_depsNoOp'^gradients/dropout_1/mul_1_grad/Reshape)^gradients/dropout_1/mul_1_grad/Reshape_1
?
7gradients/dropout_1/mul_1_grad/tuple/control_dependencyIdentity&gradients/dropout_1/mul_1_grad/Reshape0^gradients/dropout_1/mul_1_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/dropout_1/mul_1_grad/Reshape*
_output_shapes
:
?
9gradients/dropout_1/mul_1_grad/tuple/control_dependency_1Identity(gradients/dropout_1/mul_1_grad/Reshape_10^gradients/dropout_1/mul_1_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/dropout_1/mul_1_grad/Reshape_1*
_output_shapes
:
h
"gradients/dropout_1/mul_grad/ShapeShapeRelu_2*
T0*
out_type0*
_output_shapes
:
~
$gradients/dropout_1/mul_grad/Shape_1Shapedropout_1/truediv*
T0*
out_type0*#
_output_shapes
:?????????
?
2gradients/dropout_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/dropout_1/mul_grad/Shape$gradients/dropout_1/mul_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
 gradients/dropout_1/mul_grad/MulMul7gradients/dropout_1/mul_1_grad/tuple/control_dependencydropout_1/truediv*
_output_shapes
:*
T0
?
 gradients/dropout_1/mul_grad/SumSum gradients/dropout_1/mul_grad/Mul2gradients/dropout_1/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
$gradients/dropout_1/mul_grad/ReshapeReshape gradients/dropout_1/mul_grad/Sum"gradients/dropout_1/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
"gradients/dropout_1/mul_grad/Mul_1MulRelu_27gradients/dropout_1/mul_1_grad/tuple/control_dependency*
T0*
_output_shapes
:
?
"gradients/dropout_1/mul_grad/Sum_1Sum"gradients/dropout_1/mul_grad/Mul_14gradients/dropout_1/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
&gradients/dropout_1/mul_grad/Reshape_1Reshape"gradients/dropout_1/mul_grad/Sum_1$gradients/dropout_1/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
?
-gradients/dropout_1/mul_grad/tuple/group_depsNoOp%^gradients/dropout_1/mul_grad/Reshape'^gradients/dropout_1/mul_grad/Reshape_1
?
5gradients/dropout_1/mul_grad/tuple/control_dependencyIdentity$gradients/dropout_1/mul_grad/Reshape.^gradients/dropout_1/mul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dropout_1/mul_grad/Reshape*'
_output_shapes
:?????????
?
7gradients/dropout_1/mul_grad/tuple/control_dependency_1Identity&gradients/dropout_1/mul_grad/Reshape_1.^gradients/dropout_1/mul_grad/tuple/group_deps*9
_class/
-+loc:@gradients/dropout_1/mul_grad/Reshape_1*
_output_shapes
:*
T0
?
gradients/Relu_2_grad/ReluGradReluGrad5gradients/dropout_1/mul_grad/tuple/control_dependencyRelu_2*'
_output_shapes
:?????????*
T0
b
gradients/add_4_grad/ShapeShapeMatMul_2*
T0*
out_type0*
_output_shapes
:
f
gradients/add_4_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
?
*gradients/add_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_4_grad/Shapegradients/add_4_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients/add_4_grad/SumSumgradients/Relu_2_grad/ReluGrad*gradients/add_4_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
gradients/add_4_grad/ReshapeReshapegradients/add_4_grad/Sumgradients/add_4_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
gradients/add_4_grad/Sum_1Sumgradients/Relu_2_grad/ReluGrad,gradients/add_4_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
gradients/add_4_grad/Reshape_1Reshapegradients/add_4_grad/Sum_1gradients/add_4_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
m
%gradients/add_4_grad/tuple/group_depsNoOp^gradients/add_4_grad/Reshape^gradients/add_4_grad/Reshape_1
?
-gradients/add_4_grad/tuple/control_dependencyIdentitygradients/add_4_grad/Reshape&^gradients/add_4_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_4_grad/Reshape*'
_output_shapes
:?????????
?
/gradients/add_4_grad/tuple/control_dependency_1Identitygradients/add_4_grad/Reshape_1&^gradients/add_4_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_4_grad/Reshape_1*
_output_shapes
:
?
gradients/MatMul_2_grad/MatMulMatMul-gradients/add_4_grad/tuple/control_dependencyVariable_7/read*(
_output_shapes
:??????????*
transpose_a( *
transpose_b(*
T0
?
 gradients/MatMul_2_grad/MatMul_1MatMul	Reshape_1-gradients/add_4_grad/tuple/control_dependency*
_output_shapes
:	?*
transpose_a(*
transpose_b( *
T0
t
(gradients/MatMul_2_grad/tuple/group_depsNoOp^gradients/MatMul_2_grad/MatMul!^gradients/MatMul_2_grad/MatMul_1
?
0gradients/MatMul_2_grad/tuple/control_dependencyIdentitygradients/MatMul_2_grad/MatMul)^gradients/MatMul_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_2_grad/MatMul*(
_output_shapes
:??????????
?
2gradients/MatMul_2_grad/tuple/control_dependency_1Identity gradients/MatMul_2_grad/MatMul_1)^gradients/MatMul_2_grad/tuple/group_deps*
_output_shapes
:	?*
T0*3
_class)
'%loc:@gradients/MatMul_2_grad/MatMul_1
g
gradients/Reshape_1_grad/ShapeShape	MaxPool_1*
T0*
out_type0*
_output_shapes
:
?
 gradients/Reshape_1_grad/ReshapeReshape0gradients/MatMul_2_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*
T0*
Tshape0*/
_output_shapes
:?????????@
?
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_1 gradients/Reshape_1_grad/Reshape*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:?????????@
?
gradients/Relu_1_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_1*
T0*/
_output_shapes
:?????????@
b
gradients/add_3_grad/ShapeShapeConv2D_1*
T0*
out_type0*
_output_shapes
:
f
gradients/add_3_grad/Shape_1Const*
valueB:@*
dtype0*
_output_shapes
:
?
*gradients/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_3_grad/Shapegradients/add_3_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients/add_3_grad/SumSumgradients/Relu_1_grad/ReluGrad*gradients/add_3_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
gradients/add_3_grad/ReshapeReshapegradients/add_3_grad/Sumgradients/add_3_grad/Shape*
T0*
Tshape0*/
_output_shapes
:?????????@
?
gradients/add_3_grad/Sum_1Sumgradients/Relu_1_grad/ReluGrad,gradients/add_3_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
gradients/add_3_grad/Reshape_1Reshapegradients/add_3_grad/Sum_1gradients/add_3_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:@
m
%gradients/add_3_grad/tuple/group_depsNoOp^gradients/add_3_grad/Reshape^gradients/add_3_grad/Reshape_1
?
-gradients/add_3_grad/tuple/control_dependencyIdentitygradients/add_3_grad/Reshape&^gradients/add_3_grad/tuple/group_deps*/
_output_shapes
:?????????@*
T0*/
_class%
#!loc:@gradients/add_3_grad/Reshape
?
/gradients/add_3_grad/tuple/control_dependency_1Identitygradients/add_3_grad/Reshape_1&^gradients/add_3_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_3_grad/Reshape_1*
_output_shapes
:@
?
gradients/Conv2D_1_grad/ShapeNShapeNMaxPoolVariable_5/read* 
_output_shapes
::*
T0*
out_type0*
N
?
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_5/read-gradients/add_3_grad/tuple/control_dependency*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:????????? *
	dilations
*
T0*
strides
*
data_formatNHWC
?
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool gradients/Conv2D_1_grad/ShapeN:1-gradients/add_3_grad/tuple/control_dependency*
paddingSAME*&
_output_shapes
: @*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 
?
(gradients/Conv2D_1_grad/tuple/group_depsNoOp-^gradients/Conv2D_1_grad/Conv2DBackpropFilter,^gradients/Conv2D_1_grad/Conv2DBackpropInput
?
0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput*/
_output_shapes
:????????? 
?
2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*&
_output_shapes
: @*
T0*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter
?
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool0gradients/Conv2D_1_grad/tuple/control_dependency*
ksize
*
paddingSAME*/
_output_shapes
:????????? *
T0*
strides
*
data_formatNHWC
?
gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*
T0*/
_output_shapes
:????????? 
`
gradients/add_2_grad/ShapeShapeConv2D*
T0*
out_type0*
_output_shapes
:
f
gradients/add_2_grad/Shape_1Const*
_output_shapes
:*
valueB: *
dtype0
?
*gradients/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_2_grad/Shapegradients/add_2_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients/add_2_grad/SumSumgradients/Relu_grad/ReluGrad*gradients/add_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
gradients/add_2_grad/ReshapeReshapegradients/add_2_grad/Sumgradients/add_2_grad/Shape*
T0*
Tshape0*/
_output_shapes
:????????? 
?
gradients/add_2_grad/Sum_1Sumgradients/Relu_grad/ReluGrad,gradients/add_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
gradients/add_2_grad/Reshape_1Reshapegradients/add_2_grad/Sum_1gradients/add_2_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
m
%gradients/add_2_grad/tuple/group_depsNoOp^gradients/add_2_grad/Reshape^gradients/add_2_grad/Reshape_1
?
-gradients/add_2_grad/tuple/control_dependencyIdentitygradients/add_2_grad/Reshape&^gradients/add_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_2_grad/Reshape*/
_output_shapes
:????????? 
?
/gradients/add_2_grad/tuple/control_dependency_1Identitygradients/add_2_grad/Reshape_1&^gradients/add_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_2_grad/Reshape_1*
_output_shapes
: 
?
gradients/Conv2D_grad/ShapeNShapeN'SpatialTransformer/_transform/Reshape_5Variable_3/read*
N* 
_output_shapes
::*
T0*
out_type0
?
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable_3/read-gradients/add_2_grad/tuple/control_dependency*
paddingSAME*8
_output_shapes&
$:"??????????????????*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(
?
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter'SpatialTransformer/_transform/Reshape_5gradients/Conv2D_grad/ShapeN:1-gradients/add_2_grad/tuple/control_dependency*
paddingSAME*&
_output_shapes
: *
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(
?
&gradients/Conv2D_grad/tuple/group_depsNoOp+^gradients/Conv2D_grad/Conv2DBackpropFilter*^gradients/Conv2D_grad/Conv2DBackpropInput
?
.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput*8
_output_shapes&
$:"??????????????????
?
0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*&
_output_shapes
: *
T0*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter
?
<gradients/SpatialTransformer/_transform/Reshape_5_grad/ShapeShape/SpatialTransformer/_transform/_interpolate/AddN*
T0*
out_type0*
_output_shapes
:
?
>gradients/SpatialTransformer/_transform/Reshape_5_grad/ReshapeReshape.gradients/Conv2D_grad/tuple/control_dependency<gradients/SpatialTransformer/_transform/Reshape_5_grad/Shape*
T0*
Tshape0*0
_output_shapes
:??????????????????
?
Ogradients/SpatialTransformer/_transform/_interpolate/AddN_grad/tuple/group_depsNoOp?^gradients/SpatialTransformer/_transform/Reshape_5_grad/Reshape
?
Wgradients/SpatialTransformer/_transform/_interpolate/AddN_grad/tuple/control_dependencyIdentity>gradients/SpatialTransformer/_transform/Reshape_5_grad/ReshapeP^gradients/SpatialTransformer/_transform/_interpolate/AddN_grad/tuple/group_deps*0
_output_shapes
:??????????????????*
T0*Q
_classG
ECloc:@gradients/SpatialTransformer/_transform/Reshape_5_grad/Reshape
?
Ygradients/SpatialTransformer/_transform/_interpolate/AddN_grad/tuple/control_dependency_1Identity>gradients/SpatialTransformer/_transform/Reshape_5_grad/ReshapeP^gradients/SpatialTransformer/_transform/_interpolate/AddN_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/SpatialTransformer/_transform/Reshape_5_grad/Reshape*0
_output_shapes
:??????????????????
?
Ygradients/SpatialTransformer/_transform/_interpolate/AddN_grad/tuple/control_dependency_2Identity>gradients/SpatialTransformer/_transform/Reshape_5_grad/ReshapeP^gradients/SpatialTransformer/_transform/_interpolate/AddN_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/SpatialTransformer/_transform/Reshape_5_grad/Reshape*0
_output_shapes
:??????????????????
?
Ygradients/SpatialTransformer/_transform/_interpolate/AddN_grad/tuple/control_dependency_3Identity>gradients/SpatialTransformer/_transform/Reshape_5_grad/ReshapeP^gradients/SpatialTransformer/_transform/_interpolate/AddN_grad/tuple/group_deps*0
_output_shapes
:??????????????????*
T0*Q
_classG
ECloc:@gradients/SpatialTransformer/_transform/Reshape_5_grad/Reshape
?
Fgradients/SpatialTransformer/_transform/_interpolate/mul_10_grad/ShapeShape5SpatialTransformer/_transform/_interpolate/ExpandDims*
T0*
out_type0*
_output_shapes
:
?
Hgradients/SpatialTransformer/_transform/_interpolate/mul_10_grad/Shape_1Shape3SpatialTransformer/_transform/_interpolate/GatherV2*
T0*
out_type0*
_output_shapes
:
?
Vgradients/SpatialTransformer/_transform/_interpolate/mul_10_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients/SpatialTransformer/_transform/_interpolate/mul_10_grad/ShapeHgradients/SpatialTransformer/_transform/_interpolate/mul_10_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
Dgradients/SpatialTransformer/_transform/_interpolate/mul_10_grad/MulMulWgradients/SpatialTransformer/_transform/_interpolate/AddN_grad/tuple/control_dependency3SpatialTransformer/_transform/_interpolate/GatherV2*0
_output_shapes
:??????????????????*
T0
?
Dgradients/SpatialTransformer/_transform/_interpolate/mul_10_grad/SumSumDgradients/SpatialTransformer/_transform/_interpolate/mul_10_grad/MulVgradients/SpatialTransformer/_transform/_interpolate/mul_10_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
Hgradients/SpatialTransformer/_transform/_interpolate/mul_10_grad/ReshapeReshapeDgradients/SpatialTransformer/_transform/_interpolate/mul_10_grad/SumFgradients/SpatialTransformer/_transform/_interpolate/mul_10_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
Fgradients/SpatialTransformer/_transform/_interpolate/mul_10_grad/Mul_1Mul5SpatialTransformer/_transform/_interpolate/ExpandDimsWgradients/SpatialTransformer/_transform/_interpolate/AddN_grad/tuple/control_dependency*
T0*0
_output_shapes
:??????????????????
?
Fgradients/SpatialTransformer/_transform/_interpolate/mul_10_grad/Sum_1SumFgradients/SpatialTransformer/_transform/_interpolate/mul_10_grad/Mul_1Xgradients/SpatialTransformer/_transform/_interpolate/mul_10_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
Jgradients/SpatialTransformer/_transform/_interpolate/mul_10_grad/Reshape_1ReshapeFgradients/SpatialTransformer/_transform/_interpolate/mul_10_grad/Sum_1Hgradients/SpatialTransformer/_transform/_interpolate/mul_10_grad/Shape_1*
T0*
Tshape0*0
_output_shapes
:??????????????????
?
Qgradients/SpatialTransformer/_transform/_interpolate/mul_10_grad/tuple/group_depsNoOpI^gradients/SpatialTransformer/_transform/_interpolate/mul_10_grad/ReshapeK^gradients/SpatialTransformer/_transform/_interpolate/mul_10_grad/Reshape_1
?
Ygradients/SpatialTransformer/_transform/_interpolate/mul_10_grad/tuple/control_dependencyIdentityHgradients/SpatialTransformer/_transform/_interpolate/mul_10_grad/ReshapeR^gradients/SpatialTransformer/_transform/_interpolate/mul_10_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/SpatialTransformer/_transform/_interpolate/mul_10_grad/Reshape*'
_output_shapes
:?????????
?
[gradients/SpatialTransformer/_transform/_interpolate/mul_10_grad/tuple/control_dependency_1IdentityJgradients/SpatialTransformer/_transform/_interpolate/mul_10_grad/Reshape_1R^gradients/SpatialTransformer/_transform/_interpolate/mul_10_grad/tuple/group_deps*]
_classS
QOloc:@gradients/SpatialTransformer/_transform/_interpolate/mul_10_grad/Reshape_1*0
_output_shapes
:??????????????????*
T0
?
Fgradients/SpatialTransformer/_transform/_interpolate/mul_11_grad/ShapeShape7SpatialTransformer/_transform/_interpolate/ExpandDims_1*
_output_shapes
:*
T0*
out_type0
?
Hgradients/SpatialTransformer/_transform/_interpolate/mul_11_grad/Shape_1Shape5SpatialTransformer/_transform/_interpolate/GatherV2_1*
T0*
out_type0*
_output_shapes
:
?
Vgradients/SpatialTransformer/_transform/_interpolate/mul_11_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients/SpatialTransformer/_transform/_interpolate/mul_11_grad/ShapeHgradients/SpatialTransformer/_transform/_interpolate/mul_11_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
Dgradients/SpatialTransformer/_transform/_interpolate/mul_11_grad/MulMulYgradients/SpatialTransformer/_transform/_interpolate/AddN_grad/tuple/control_dependency_15SpatialTransformer/_transform/_interpolate/GatherV2_1*
T0*0
_output_shapes
:??????????????????
?
Dgradients/SpatialTransformer/_transform/_interpolate/mul_11_grad/SumSumDgradients/SpatialTransformer/_transform/_interpolate/mul_11_grad/MulVgradients/SpatialTransformer/_transform/_interpolate/mul_11_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
?
Hgradients/SpatialTransformer/_transform/_interpolate/mul_11_grad/ReshapeReshapeDgradients/SpatialTransformer/_transform/_interpolate/mul_11_grad/SumFgradients/SpatialTransformer/_transform/_interpolate/mul_11_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
Fgradients/SpatialTransformer/_transform/_interpolate/mul_11_grad/Mul_1Mul7SpatialTransformer/_transform/_interpolate/ExpandDims_1Ygradients/SpatialTransformer/_transform/_interpolate/AddN_grad/tuple/control_dependency_1*0
_output_shapes
:??????????????????*
T0
?
Fgradients/SpatialTransformer/_transform/_interpolate/mul_11_grad/Sum_1SumFgradients/SpatialTransformer/_transform/_interpolate/mul_11_grad/Mul_1Xgradients/SpatialTransformer/_transform/_interpolate/mul_11_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
?
Jgradients/SpatialTransformer/_transform/_interpolate/mul_11_grad/Reshape_1ReshapeFgradients/SpatialTransformer/_transform/_interpolate/mul_11_grad/Sum_1Hgradients/SpatialTransformer/_transform/_interpolate/mul_11_grad/Shape_1*
T0*
Tshape0*0
_output_shapes
:??????????????????
?
Qgradients/SpatialTransformer/_transform/_interpolate/mul_11_grad/tuple/group_depsNoOpI^gradients/SpatialTransformer/_transform/_interpolate/mul_11_grad/ReshapeK^gradients/SpatialTransformer/_transform/_interpolate/mul_11_grad/Reshape_1
?
Ygradients/SpatialTransformer/_transform/_interpolate/mul_11_grad/tuple/control_dependencyIdentityHgradients/SpatialTransformer/_transform/_interpolate/mul_11_grad/ReshapeR^gradients/SpatialTransformer/_transform/_interpolate/mul_11_grad/tuple/group_deps*'
_output_shapes
:?????????*
T0*[
_classQ
OMloc:@gradients/SpatialTransformer/_transform/_interpolate/mul_11_grad/Reshape
?
[gradients/SpatialTransformer/_transform/_interpolate/mul_11_grad/tuple/control_dependency_1IdentityJgradients/SpatialTransformer/_transform/_interpolate/mul_11_grad/Reshape_1R^gradients/SpatialTransformer/_transform/_interpolate/mul_11_grad/tuple/group_deps*0
_output_shapes
:??????????????????*
T0*]
_classS
QOloc:@gradients/SpatialTransformer/_transform/_interpolate/mul_11_grad/Reshape_1
?
Fgradients/SpatialTransformer/_transform/_interpolate/mul_12_grad/ShapeShape7SpatialTransformer/_transform/_interpolate/ExpandDims_2*
_output_shapes
:*
T0*
out_type0
?
Hgradients/SpatialTransformer/_transform/_interpolate/mul_12_grad/Shape_1Shape5SpatialTransformer/_transform/_interpolate/GatherV2_2*
out_type0*
_output_shapes
:*
T0
?
Vgradients/SpatialTransformer/_transform/_interpolate/mul_12_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients/SpatialTransformer/_transform/_interpolate/mul_12_grad/ShapeHgradients/SpatialTransformer/_transform/_interpolate/mul_12_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
Dgradients/SpatialTransformer/_transform/_interpolate/mul_12_grad/MulMulYgradients/SpatialTransformer/_transform/_interpolate/AddN_grad/tuple/control_dependency_25SpatialTransformer/_transform/_interpolate/GatherV2_2*0
_output_shapes
:??????????????????*
T0
?
Dgradients/SpatialTransformer/_transform/_interpolate/mul_12_grad/SumSumDgradients/SpatialTransformer/_transform/_interpolate/mul_12_grad/MulVgradients/SpatialTransformer/_transform/_interpolate/mul_12_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
Hgradients/SpatialTransformer/_transform/_interpolate/mul_12_grad/ReshapeReshapeDgradients/SpatialTransformer/_transform/_interpolate/mul_12_grad/SumFgradients/SpatialTransformer/_transform/_interpolate/mul_12_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
Fgradients/SpatialTransformer/_transform/_interpolate/mul_12_grad/Mul_1Mul7SpatialTransformer/_transform/_interpolate/ExpandDims_2Ygradients/SpatialTransformer/_transform/_interpolate/AddN_grad/tuple/control_dependency_2*
T0*0
_output_shapes
:??????????????????
?
Fgradients/SpatialTransformer/_transform/_interpolate/mul_12_grad/Sum_1SumFgradients/SpatialTransformer/_transform/_interpolate/mul_12_grad/Mul_1Xgradients/SpatialTransformer/_transform/_interpolate/mul_12_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
Jgradients/SpatialTransformer/_transform/_interpolate/mul_12_grad/Reshape_1ReshapeFgradients/SpatialTransformer/_transform/_interpolate/mul_12_grad/Sum_1Hgradients/SpatialTransformer/_transform/_interpolate/mul_12_grad/Shape_1*
Tshape0*0
_output_shapes
:??????????????????*
T0
?
Qgradients/SpatialTransformer/_transform/_interpolate/mul_12_grad/tuple/group_depsNoOpI^gradients/SpatialTransformer/_transform/_interpolate/mul_12_grad/ReshapeK^gradients/SpatialTransformer/_transform/_interpolate/mul_12_grad/Reshape_1
?
Ygradients/SpatialTransformer/_transform/_interpolate/mul_12_grad/tuple/control_dependencyIdentityHgradients/SpatialTransformer/_transform/_interpolate/mul_12_grad/ReshapeR^gradients/SpatialTransformer/_transform/_interpolate/mul_12_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/SpatialTransformer/_transform/_interpolate/mul_12_grad/Reshape*'
_output_shapes
:?????????
?
[gradients/SpatialTransformer/_transform/_interpolate/mul_12_grad/tuple/control_dependency_1IdentityJgradients/SpatialTransformer/_transform/_interpolate/mul_12_grad/Reshape_1R^gradients/SpatialTransformer/_transform/_interpolate/mul_12_grad/tuple/group_deps*0
_output_shapes
:??????????????????*
T0*]
_classS
QOloc:@gradients/SpatialTransformer/_transform/_interpolate/mul_12_grad/Reshape_1
?
Fgradients/SpatialTransformer/_transform/_interpolate/mul_13_grad/ShapeShape7SpatialTransformer/_transform/_interpolate/ExpandDims_3*
_output_shapes
:*
T0*
out_type0
?
Hgradients/SpatialTransformer/_transform/_interpolate/mul_13_grad/Shape_1Shape5SpatialTransformer/_transform/_interpolate/GatherV2_3*
_output_shapes
:*
T0*
out_type0
?
Vgradients/SpatialTransformer/_transform/_interpolate/mul_13_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients/SpatialTransformer/_transform/_interpolate/mul_13_grad/ShapeHgradients/SpatialTransformer/_transform/_interpolate/mul_13_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
Dgradients/SpatialTransformer/_transform/_interpolate/mul_13_grad/MulMulYgradients/SpatialTransformer/_transform/_interpolate/AddN_grad/tuple/control_dependency_35SpatialTransformer/_transform/_interpolate/GatherV2_3*
T0*0
_output_shapes
:??????????????????
?
Dgradients/SpatialTransformer/_transform/_interpolate/mul_13_grad/SumSumDgradients/SpatialTransformer/_transform/_interpolate/mul_13_grad/MulVgradients/SpatialTransformer/_transform/_interpolate/mul_13_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
Hgradients/SpatialTransformer/_transform/_interpolate/mul_13_grad/ReshapeReshapeDgradients/SpatialTransformer/_transform/_interpolate/mul_13_grad/SumFgradients/SpatialTransformer/_transform/_interpolate/mul_13_grad/Shape*'
_output_shapes
:?????????*
T0*
Tshape0
?
Fgradients/SpatialTransformer/_transform/_interpolate/mul_13_grad/Mul_1Mul7SpatialTransformer/_transform/_interpolate/ExpandDims_3Ygradients/SpatialTransformer/_transform/_interpolate/AddN_grad/tuple/control_dependency_3*0
_output_shapes
:??????????????????*
T0
?
Fgradients/SpatialTransformer/_transform/_interpolate/mul_13_grad/Sum_1SumFgradients/SpatialTransformer/_transform/_interpolate/mul_13_grad/Mul_1Xgradients/SpatialTransformer/_transform/_interpolate/mul_13_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
Jgradients/SpatialTransformer/_transform/_interpolate/mul_13_grad/Reshape_1ReshapeFgradients/SpatialTransformer/_transform/_interpolate/mul_13_grad/Sum_1Hgradients/SpatialTransformer/_transform/_interpolate/mul_13_grad/Shape_1*
T0*
Tshape0*0
_output_shapes
:??????????????????
?
Qgradients/SpatialTransformer/_transform/_interpolate/mul_13_grad/tuple/group_depsNoOpI^gradients/SpatialTransformer/_transform/_interpolate/mul_13_grad/ReshapeK^gradients/SpatialTransformer/_transform/_interpolate/mul_13_grad/Reshape_1
?
Ygradients/SpatialTransformer/_transform/_interpolate/mul_13_grad/tuple/control_dependencyIdentityHgradients/SpatialTransformer/_transform/_interpolate/mul_13_grad/ReshapeR^gradients/SpatialTransformer/_transform/_interpolate/mul_13_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/SpatialTransformer/_transform/_interpolate/mul_13_grad/Reshape*'
_output_shapes
:?????????
?
[gradients/SpatialTransformer/_transform/_interpolate/mul_13_grad/tuple/control_dependency_1IdentityJgradients/SpatialTransformer/_transform/_interpolate/mul_13_grad/Reshape_1R^gradients/SpatialTransformer/_transform/_interpolate/mul_13_grad/tuple/group_deps*0
_output_shapes
:??????????????????*
T0*]
_classS
QOloc:@gradients/SpatialTransformer/_transform/_interpolate/mul_13_grad/Reshape_1
?
Jgradients/SpatialTransformer/_transform/_interpolate/ExpandDims_grad/ShapeShape0SpatialTransformer/_transform/_interpolate/mul_6*
T0*
out_type0*
_output_shapes
:
?
Lgradients/SpatialTransformer/_transform/_interpolate/ExpandDims_grad/ReshapeReshapeYgradients/SpatialTransformer/_transform/_interpolate/mul_10_grad/tuple/control_dependencyJgradients/SpatialTransformer/_transform/_interpolate/ExpandDims_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
Lgradients/SpatialTransformer/_transform/_interpolate/ExpandDims_1_grad/ShapeShape0SpatialTransformer/_transform/_interpolate/mul_7*
T0*
out_type0*
_output_shapes
:
?
Ngradients/SpatialTransformer/_transform/_interpolate/ExpandDims_1_grad/ReshapeReshapeYgradients/SpatialTransformer/_transform/_interpolate/mul_11_grad/tuple/control_dependencyLgradients/SpatialTransformer/_transform/_interpolate/ExpandDims_1_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
Lgradients/SpatialTransformer/_transform/_interpolate/ExpandDims_2_grad/ShapeShape0SpatialTransformer/_transform/_interpolate/mul_8*
T0*
out_type0*
_output_shapes
:
?
Ngradients/SpatialTransformer/_transform/_interpolate/ExpandDims_2_grad/ReshapeReshapeYgradients/SpatialTransformer/_transform/_interpolate/mul_12_grad/tuple/control_dependencyLgradients/SpatialTransformer/_transform/_interpolate/ExpandDims_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
Lgradients/SpatialTransformer/_transform/_interpolate/ExpandDims_3_grad/ShapeShape0SpatialTransformer/_transform/_interpolate/mul_9*
_output_shapes
:*
T0*
out_type0
?
Ngradients/SpatialTransformer/_transform/_interpolate/ExpandDims_3_grad/ReshapeReshapeYgradients/SpatialTransformer/_transform/_interpolate/mul_13_grad/tuple/control_dependencyLgradients/SpatialTransformer/_transform/_interpolate/ExpandDims_3_grad/Shape*
Tshape0*#
_output_shapes
:?????????*
T0
?
Egradients/SpatialTransformer/_transform/_interpolate/mul_6_grad/ShapeShape0SpatialTransformer/_transform/_interpolate/sub_2*
_output_shapes
:*
T0*
out_type0
?
Ggradients/SpatialTransformer/_transform/_interpolate/mul_6_grad/Shape_1Shape0SpatialTransformer/_transform/_interpolate/sub_3*
out_type0*
_output_shapes
:*
T0
?
Ugradients/SpatialTransformer/_transform/_interpolate/mul_6_grad/BroadcastGradientArgsBroadcastGradientArgsEgradients/SpatialTransformer/_transform/_interpolate/mul_6_grad/ShapeGgradients/SpatialTransformer/_transform/_interpolate/mul_6_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
Cgradients/SpatialTransformer/_transform/_interpolate/mul_6_grad/MulMulLgradients/SpatialTransformer/_transform/_interpolate/ExpandDims_grad/Reshape0SpatialTransformer/_transform/_interpolate/sub_3*#
_output_shapes
:?????????*
T0
?
Cgradients/SpatialTransformer/_transform/_interpolate/mul_6_grad/SumSumCgradients/SpatialTransformer/_transform/_interpolate/mul_6_grad/MulUgradients/SpatialTransformer/_transform/_interpolate/mul_6_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
?
Ggradients/SpatialTransformer/_transform/_interpolate/mul_6_grad/ReshapeReshapeCgradients/SpatialTransformer/_transform/_interpolate/mul_6_grad/SumEgradients/SpatialTransformer/_transform/_interpolate/mul_6_grad/Shape*#
_output_shapes
:?????????*
T0*
Tshape0
?
Egradients/SpatialTransformer/_transform/_interpolate/mul_6_grad/Mul_1Mul0SpatialTransformer/_transform/_interpolate/sub_2Lgradients/SpatialTransformer/_transform/_interpolate/ExpandDims_grad/Reshape*#
_output_shapes
:?????????*
T0
?
Egradients/SpatialTransformer/_transform/_interpolate/mul_6_grad/Sum_1SumEgradients/SpatialTransformer/_transform/_interpolate/mul_6_grad/Mul_1Wgradients/SpatialTransformer/_transform/_interpolate/mul_6_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
Igradients/SpatialTransformer/_transform/_interpolate/mul_6_grad/Reshape_1ReshapeEgradients/SpatialTransformer/_transform/_interpolate/mul_6_grad/Sum_1Ggradients/SpatialTransformer/_transform/_interpolate/mul_6_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:?????????
?
Pgradients/SpatialTransformer/_transform/_interpolate/mul_6_grad/tuple/group_depsNoOpH^gradients/SpatialTransformer/_transform/_interpolate/mul_6_grad/ReshapeJ^gradients/SpatialTransformer/_transform/_interpolate/mul_6_grad/Reshape_1
?
Xgradients/SpatialTransformer/_transform/_interpolate/mul_6_grad/tuple/control_dependencyIdentityGgradients/SpatialTransformer/_transform/_interpolate/mul_6_grad/ReshapeQ^gradients/SpatialTransformer/_transform/_interpolate/mul_6_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients/SpatialTransformer/_transform/_interpolate/mul_6_grad/Reshape*#
_output_shapes
:?????????
?
Zgradients/SpatialTransformer/_transform/_interpolate/mul_6_grad/tuple/control_dependency_1IdentityIgradients/SpatialTransformer/_transform/_interpolate/mul_6_grad/Reshape_1Q^gradients/SpatialTransformer/_transform/_interpolate/mul_6_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/SpatialTransformer/_transform/_interpolate/mul_6_grad/Reshape_1*#
_output_shapes
:?????????
?
Egradients/SpatialTransformer/_transform/_interpolate/mul_7_grad/ShapeShape0SpatialTransformer/_transform/_interpolate/sub_4*
T0*
out_type0*
_output_shapes
:
?
Ggradients/SpatialTransformer/_transform/_interpolate/mul_7_grad/Shape_1Shape0SpatialTransformer/_transform/_interpolate/sub_5*
_output_shapes
:*
T0*
out_type0
?
Ugradients/SpatialTransformer/_transform/_interpolate/mul_7_grad/BroadcastGradientArgsBroadcastGradientArgsEgradients/SpatialTransformer/_transform/_interpolate/mul_7_grad/ShapeGgradients/SpatialTransformer/_transform/_interpolate/mul_7_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
Cgradients/SpatialTransformer/_transform/_interpolate/mul_7_grad/MulMulNgradients/SpatialTransformer/_transform/_interpolate/ExpandDims_1_grad/Reshape0SpatialTransformer/_transform/_interpolate/sub_5*
T0*#
_output_shapes
:?????????
?
Cgradients/SpatialTransformer/_transform/_interpolate/mul_7_grad/SumSumCgradients/SpatialTransformer/_transform/_interpolate/mul_7_grad/MulUgradients/SpatialTransformer/_transform/_interpolate/mul_7_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
Ggradients/SpatialTransformer/_transform/_interpolate/mul_7_grad/ReshapeReshapeCgradients/SpatialTransformer/_transform/_interpolate/mul_7_grad/SumEgradients/SpatialTransformer/_transform/_interpolate/mul_7_grad/Shape*#
_output_shapes
:?????????*
T0*
Tshape0
?
Egradients/SpatialTransformer/_transform/_interpolate/mul_7_grad/Mul_1Mul0SpatialTransformer/_transform/_interpolate/sub_4Ngradients/SpatialTransformer/_transform/_interpolate/ExpandDims_1_grad/Reshape*
T0*#
_output_shapes
:?????????
?
Egradients/SpatialTransformer/_transform/_interpolate/mul_7_grad/Sum_1SumEgradients/SpatialTransformer/_transform/_interpolate/mul_7_grad/Mul_1Wgradients/SpatialTransformer/_transform/_interpolate/mul_7_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
?
Igradients/SpatialTransformer/_transform/_interpolate/mul_7_grad/Reshape_1ReshapeEgradients/SpatialTransformer/_transform/_interpolate/mul_7_grad/Sum_1Ggradients/SpatialTransformer/_transform/_interpolate/mul_7_grad/Shape_1*#
_output_shapes
:?????????*
T0*
Tshape0
?
Pgradients/SpatialTransformer/_transform/_interpolate/mul_7_grad/tuple/group_depsNoOpH^gradients/SpatialTransformer/_transform/_interpolate/mul_7_grad/ReshapeJ^gradients/SpatialTransformer/_transform/_interpolate/mul_7_grad/Reshape_1
?
Xgradients/SpatialTransformer/_transform/_interpolate/mul_7_grad/tuple/control_dependencyIdentityGgradients/SpatialTransformer/_transform/_interpolate/mul_7_grad/ReshapeQ^gradients/SpatialTransformer/_transform/_interpolate/mul_7_grad/tuple/group_deps*#
_output_shapes
:?????????*
T0*Z
_classP
NLloc:@gradients/SpatialTransformer/_transform/_interpolate/mul_7_grad/Reshape
?
Zgradients/SpatialTransformer/_transform/_interpolate/mul_7_grad/tuple/control_dependency_1IdentityIgradients/SpatialTransformer/_transform/_interpolate/mul_7_grad/Reshape_1Q^gradients/SpatialTransformer/_transform/_interpolate/mul_7_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/SpatialTransformer/_transform/_interpolate/mul_7_grad/Reshape_1*#
_output_shapes
:?????????
?
Egradients/SpatialTransformer/_transform/_interpolate/mul_8_grad/ShapeShape0SpatialTransformer/_transform/_interpolate/sub_6*
_output_shapes
:*
T0*
out_type0
?
Ggradients/SpatialTransformer/_transform/_interpolate/mul_8_grad/Shape_1Shape0SpatialTransformer/_transform/_interpolate/sub_7*
_output_shapes
:*
T0*
out_type0
?
Ugradients/SpatialTransformer/_transform/_interpolate/mul_8_grad/BroadcastGradientArgsBroadcastGradientArgsEgradients/SpatialTransformer/_transform/_interpolate/mul_8_grad/ShapeGgradients/SpatialTransformer/_transform/_interpolate/mul_8_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
Cgradients/SpatialTransformer/_transform/_interpolate/mul_8_grad/MulMulNgradients/SpatialTransformer/_transform/_interpolate/ExpandDims_2_grad/Reshape0SpatialTransformer/_transform/_interpolate/sub_7*
T0*#
_output_shapes
:?????????
?
Cgradients/SpatialTransformer/_transform/_interpolate/mul_8_grad/SumSumCgradients/SpatialTransformer/_transform/_interpolate/mul_8_grad/MulUgradients/SpatialTransformer/_transform/_interpolate/mul_8_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
Ggradients/SpatialTransformer/_transform/_interpolate/mul_8_grad/ReshapeReshapeCgradients/SpatialTransformer/_transform/_interpolate/mul_8_grad/SumEgradients/SpatialTransformer/_transform/_interpolate/mul_8_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
Egradients/SpatialTransformer/_transform/_interpolate/mul_8_grad/Mul_1Mul0SpatialTransformer/_transform/_interpolate/sub_6Ngradients/SpatialTransformer/_transform/_interpolate/ExpandDims_2_grad/Reshape*
T0*#
_output_shapes
:?????????
?
Egradients/SpatialTransformer/_transform/_interpolate/mul_8_grad/Sum_1SumEgradients/SpatialTransformer/_transform/_interpolate/mul_8_grad/Mul_1Wgradients/SpatialTransformer/_transform/_interpolate/mul_8_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
Igradients/SpatialTransformer/_transform/_interpolate/mul_8_grad/Reshape_1ReshapeEgradients/SpatialTransformer/_transform/_interpolate/mul_8_grad/Sum_1Ggradients/SpatialTransformer/_transform/_interpolate/mul_8_grad/Shape_1*#
_output_shapes
:?????????*
T0*
Tshape0
?
Pgradients/SpatialTransformer/_transform/_interpolate/mul_8_grad/tuple/group_depsNoOpH^gradients/SpatialTransformer/_transform/_interpolate/mul_8_grad/ReshapeJ^gradients/SpatialTransformer/_transform/_interpolate/mul_8_grad/Reshape_1
?
Xgradients/SpatialTransformer/_transform/_interpolate/mul_8_grad/tuple/control_dependencyIdentityGgradients/SpatialTransformer/_transform/_interpolate/mul_8_grad/ReshapeQ^gradients/SpatialTransformer/_transform/_interpolate/mul_8_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients/SpatialTransformer/_transform/_interpolate/mul_8_grad/Reshape*#
_output_shapes
:?????????
?
Zgradients/SpatialTransformer/_transform/_interpolate/mul_8_grad/tuple/control_dependency_1IdentityIgradients/SpatialTransformer/_transform/_interpolate/mul_8_grad/Reshape_1Q^gradients/SpatialTransformer/_transform/_interpolate/mul_8_grad/tuple/group_deps*#
_output_shapes
:?????????*
T0*\
_classR
PNloc:@gradients/SpatialTransformer/_transform/_interpolate/mul_8_grad/Reshape_1
?
Egradients/SpatialTransformer/_transform/_interpolate/mul_9_grad/ShapeShape0SpatialTransformer/_transform/_interpolate/sub_8*
T0*
out_type0*
_output_shapes
:
?
Ggradients/SpatialTransformer/_transform/_interpolate/mul_9_grad/Shape_1Shape0SpatialTransformer/_transform/_interpolate/sub_9*
T0*
out_type0*
_output_shapes
:
?
Ugradients/SpatialTransformer/_transform/_interpolate/mul_9_grad/BroadcastGradientArgsBroadcastGradientArgsEgradients/SpatialTransformer/_transform/_interpolate/mul_9_grad/ShapeGgradients/SpatialTransformer/_transform/_interpolate/mul_9_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
Cgradients/SpatialTransformer/_transform/_interpolate/mul_9_grad/MulMulNgradients/SpatialTransformer/_transform/_interpolate/ExpandDims_3_grad/Reshape0SpatialTransformer/_transform/_interpolate/sub_9*#
_output_shapes
:?????????*
T0
?
Cgradients/SpatialTransformer/_transform/_interpolate/mul_9_grad/SumSumCgradients/SpatialTransformer/_transform/_interpolate/mul_9_grad/MulUgradients/SpatialTransformer/_transform/_interpolate/mul_9_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
?
Ggradients/SpatialTransformer/_transform/_interpolate/mul_9_grad/ReshapeReshapeCgradients/SpatialTransformer/_transform/_interpolate/mul_9_grad/SumEgradients/SpatialTransformer/_transform/_interpolate/mul_9_grad/Shape*#
_output_shapes
:?????????*
T0*
Tshape0
?
Egradients/SpatialTransformer/_transform/_interpolate/mul_9_grad/Mul_1Mul0SpatialTransformer/_transform/_interpolate/sub_8Ngradients/SpatialTransformer/_transform/_interpolate/ExpandDims_3_grad/Reshape*
T0*#
_output_shapes
:?????????
?
Egradients/SpatialTransformer/_transform/_interpolate/mul_9_grad/Sum_1SumEgradients/SpatialTransformer/_transform/_interpolate/mul_9_grad/Mul_1Wgradients/SpatialTransformer/_transform/_interpolate/mul_9_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
Igradients/SpatialTransformer/_transform/_interpolate/mul_9_grad/Reshape_1ReshapeEgradients/SpatialTransformer/_transform/_interpolate/mul_9_grad/Sum_1Ggradients/SpatialTransformer/_transform/_interpolate/mul_9_grad/Shape_1*#
_output_shapes
:?????????*
T0*
Tshape0
?
Pgradients/SpatialTransformer/_transform/_interpolate/mul_9_grad/tuple/group_depsNoOpH^gradients/SpatialTransformer/_transform/_interpolate/mul_9_grad/ReshapeJ^gradients/SpatialTransformer/_transform/_interpolate/mul_9_grad/Reshape_1
?
Xgradients/SpatialTransformer/_transform/_interpolate/mul_9_grad/tuple/control_dependencyIdentityGgradients/SpatialTransformer/_transform/_interpolate/mul_9_grad/ReshapeQ^gradients/SpatialTransformer/_transform/_interpolate/mul_9_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients/SpatialTransformer/_transform/_interpolate/mul_9_grad/Reshape*#
_output_shapes
:?????????
?
Zgradients/SpatialTransformer/_transform/_interpolate/mul_9_grad/tuple/control_dependency_1IdentityIgradients/SpatialTransformer/_transform/_interpolate/mul_9_grad/Reshape_1Q^gradients/SpatialTransformer/_transform/_interpolate/mul_9_grad/tuple/group_deps*#
_output_shapes
:?????????*
T0*\
_classR
PNloc:@gradients/SpatialTransformer/_transform/_interpolate/mul_9_grad/Reshape_1
?
Egradients/SpatialTransformer/_transform/_interpolate/sub_2_grad/ShapeShape1SpatialTransformer/_transform/_interpolate/Cast_5*
T0*
out_type0*
_output_shapes
:
?
Ggradients/SpatialTransformer/_transform/_interpolate/sub_2_grad/Shape_1Shape2SpatialTransformer/_transform/_interpolate/truediv*
T0*
out_type0*
_output_shapes
:
?
Ugradients/SpatialTransformer/_transform/_interpolate/sub_2_grad/BroadcastGradientArgsBroadcastGradientArgsEgradients/SpatialTransformer/_transform/_interpolate/sub_2_grad/ShapeGgradients/SpatialTransformer/_transform/_interpolate/sub_2_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
Cgradients/SpatialTransformer/_transform/_interpolate/sub_2_grad/SumSumXgradients/SpatialTransformer/_transform/_interpolate/mul_6_grad/tuple/control_dependencyUgradients/SpatialTransformer/_transform/_interpolate/sub_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
Ggradients/SpatialTransformer/_transform/_interpolate/sub_2_grad/ReshapeReshapeCgradients/SpatialTransformer/_transform/_interpolate/sub_2_grad/SumEgradients/SpatialTransformer/_transform/_interpolate/sub_2_grad/Shape*#
_output_shapes
:?????????*
T0*
Tshape0
?
Egradients/SpatialTransformer/_transform/_interpolate/sub_2_grad/Sum_1SumXgradients/SpatialTransformer/_transform/_interpolate/mul_6_grad/tuple/control_dependencyWgradients/SpatialTransformer/_transform/_interpolate/sub_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
?
Cgradients/SpatialTransformer/_transform/_interpolate/sub_2_grad/NegNegEgradients/SpatialTransformer/_transform/_interpolate/sub_2_grad/Sum_1*
T0*
_output_shapes
:
?
Igradients/SpatialTransformer/_transform/_interpolate/sub_2_grad/Reshape_1ReshapeCgradients/SpatialTransformer/_transform/_interpolate/sub_2_grad/NegGgradients/SpatialTransformer/_transform/_interpolate/sub_2_grad/Shape_1*#
_output_shapes
:?????????*
T0*
Tshape0
?
Pgradients/SpatialTransformer/_transform/_interpolate/sub_2_grad/tuple/group_depsNoOpH^gradients/SpatialTransformer/_transform/_interpolate/sub_2_grad/ReshapeJ^gradients/SpatialTransformer/_transform/_interpolate/sub_2_grad/Reshape_1
?
Xgradients/SpatialTransformer/_transform/_interpolate/sub_2_grad/tuple/control_dependencyIdentityGgradients/SpatialTransformer/_transform/_interpolate/sub_2_grad/ReshapeQ^gradients/SpatialTransformer/_transform/_interpolate/sub_2_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients/SpatialTransformer/_transform/_interpolate/sub_2_grad/Reshape*#
_output_shapes
:?????????
?
Zgradients/SpatialTransformer/_transform/_interpolate/sub_2_grad/tuple/control_dependency_1IdentityIgradients/SpatialTransformer/_transform/_interpolate/sub_2_grad/Reshape_1Q^gradients/SpatialTransformer/_transform/_interpolate/sub_2_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/SpatialTransformer/_transform/_interpolate/sub_2_grad/Reshape_1*#
_output_shapes
:?????????
?
Egradients/SpatialTransformer/_transform/_interpolate/sub_3_grad/ShapeShape1SpatialTransformer/_transform/_interpolate/Cast_7*
_output_shapes
:*
T0*
out_type0
?
Ggradients/SpatialTransformer/_transform/_interpolate/sub_3_grad/Shape_1Shape4SpatialTransformer/_transform/_interpolate/truediv_1*
T0*
out_type0*
_output_shapes
:
?
Ugradients/SpatialTransformer/_transform/_interpolate/sub_3_grad/BroadcastGradientArgsBroadcastGradientArgsEgradients/SpatialTransformer/_transform/_interpolate/sub_3_grad/ShapeGgradients/SpatialTransformer/_transform/_interpolate/sub_3_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
Cgradients/SpatialTransformer/_transform/_interpolate/sub_3_grad/SumSumZgradients/SpatialTransformer/_transform/_interpolate/mul_6_grad/tuple/control_dependency_1Ugradients/SpatialTransformer/_transform/_interpolate/sub_3_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
Ggradients/SpatialTransformer/_transform/_interpolate/sub_3_grad/ReshapeReshapeCgradients/SpatialTransformer/_transform/_interpolate/sub_3_grad/SumEgradients/SpatialTransformer/_transform/_interpolate/sub_3_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
Egradients/SpatialTransformer/_transform/_interpolate/sub_3_grad/Sum_1SumZgradients/SpatialTransformer/_transform/_interpolate/mul_6_grad/tuple/control_dependency_1Wgradients/SpatialTransformer/_transform/_interpolate/sub_3_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
Cgradients/SpatialTransformer/_transform/_interpolate/sub_3_grad/NegNegEgradients/SpatialTransformer/_transform/_interpolate/sub_3_grad/Sum_1*
_output_shapes
:*
T0
?
Igradients/SpatialTransformer/_transform/_interpolate/sub_3_grad/Reshape_1ReshapeCgradients/SpatialTransformer/_transform/_interpolate/sub_3_grad/NegGgradients/SpatialTransformer/_transform/_interpolate/sub_3_grad/Shape_1*#
_output_shapes
:?????????*
T0*
Tshape0
?
Pgradients/SpatialTransformer/_transform/_interpolate/sub_3_grad/tuple/group_depsNoOpH^gradients/SpatialTransformer/_transform/_interpolate/sub_3_grad/ReshapeJ^gradients/SpatialTransformer/_transform/_interpolate/sub_3_grad/Reshape_1
?
Xgradients/SpatialTransformer/_transform/_interpolate/sub_3_grad/tuple/control_dependencyIdentityGgradients/SpatialTransformer/_transform/_interpolate/sub_3_grad/ReshapeQ^gradients/SpatialTransformer/_transform/_interpolate/sub_3_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients/SpatialTransformer/_transform/_interpolate/sub_3_grad/Reshape*#
_output_shapes
:?????????
?
Zgradients/SpatialTransformer/_transform/_interpolate/sub_3_grad/tuple/control_dependency_1IdentityIgradients/SpatialTransformer/_transform/_interpolate/sub_3_grad/Reshape_1Q^gradients/SpatialTransformer/_transform/_interpolate/sub_3_grad/tuple/group_deps*#
_output_shapes
:?????????*
T0*\
_classR
PNloc:@gradients/SpatialTransformer/_transform/_interpolate/sub_3_grad/Reshape_1
?
Egradients/SpatialTransformer/_transform/_interpolate/sub_4_grad/ShapeShape1SpatialTransformer/_transform/_interpolate/Cast_5*
T0*
out_type0*
_output_shapes
:
?
Ggradients/SpatialTransformer/_transform/_interpolate/sub_4_grad/Shape_1Shape2SpatialTransformer/_transform/_interpolate/truediv*
T0*
out_type0*
_output_shapes
:
?
Ugradients/SpatialTransformer/_transform/_interpolate/sub_4_grad/BroadcastGradientArgsBroadcastGradientArgsEgradients/SpatialTransformer/_transform/_interpolate/sub_4_grad/ShapeGgradients/SpatialTransformer/_transform/_interpolate/sub_4_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
Cgradients/SpatialTransformer/_transform/_interpolate/sub_4_grad/SumSumXgradients/SpatialTransformer/_transform/_interpolate/mul_7_grad/tuple/control_dependencyUgradients/SpatialTransformer/_transform/_interpolate/sub_4_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
Ggradients/SpatialTransformer/_transform/_interpolate/sub_4_grad/ReshapeReshapeCgradients/SpatialTransformer/_transform/_interpolate/sub_4_grad/SumEgradients/SpatialTransformer/_transform/_interpolate/sub_4_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
Egradients/SpatialTransformer/_transform/_interpolate/sub_4_grad/Sum_1SumXgradients/SpatialTransformer/_transform/_interpolate/mul_7_grad/tuple/control_dependencyWgradients/SpatialTransformer/_transform/_interpolate/sub_4_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
Cgradients/SpatialTransformer/_transform/_interpolate/sub_4_grad/NegNegEgradients/SpatialTransformer/_transform/_interpolate/sub_4_grad/Sum_1*
_output_shapes
:*
T0
?
Igradients/SpatialTransformer/_transform/_interpolate/sub_4_grad/Reshape_1ReshapeCgradients/SpatialTransformer/_transform/_interpolate/sub_4_grad/NegGgradients/SpatialTransformer/_transform/_interpolate/sub_4_grad/Shape_1*#
_output_shapes
:?????????*
T0*
Tshape0
?
Pgradients/SpatialTransformer/_transform/_interpolate/sub_4_grad/tuple/group_depsNoOpH^gradients/SpatialTransformer/_transform/_interpolate/sub_4_grad/ReshapeJ^gradients/SpatialTransformer/_transform/_interpolate/sub_4_grad/Reshape_1
?
Xgradients/SpatialTransformer/_transform/_interpolate/sub_4_grad/tuple/control_dependencyIdentityGgradients/SpatialTransformer/_transform/_interpolate/sub_4_grad/ReshapeQ^gradients/SpatialTransformer/_transform/_interpolate/sub_4_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients/SpatialTransformer/_transform/_interpolate/sub_4_grad/Reshape*#
_output_shapes
:?????????
?
Zgradients/SpatialTransformer/_transform/_interpolate/sub_4_grad/tuple/control_dependency_1IdentityIgradients/SpatialTransformer/_transform/_interpolate/sub_4_grad/Reshape_1Q^gradients/SpatialTransformer/_transform/_interpolate/sub_4_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/SpatialTransformer/_transform/_interpolate/sub_4_grad/Reshape_1*#
_output_shapes
:?????????
?
Egradients/SpatialTransformer/_transform/_interpolate/sub_5_grad/ShapeShape4SpatialTransformer/_transform/_interpolate/truediv_1*
T0*
out_type0*
_output_shapes
:
?
Ggradients/SpatialTransformer/_transform/_interpolate/sub_5_grad/Shape_1Shape1SpatialTransformer/_transform/_interpolate/Cast_6*
T0*
out_type0*
_output_shapes
:
?
Ugradients/SpatialTransformer/_transform/_interpolate/sub_5_grad/BroadcastGradientArgsBroadcastGradientArgsEgradients/SpatialTransformer/_transform/_interpolate/sub_5_grad/ShapeGgradients/SpatialTransformer/_transform/_interpolate/sub_5_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
Cgradients/SpatialTransformer/_transform/_interpolate/sub_5_grad/SumSumZgradients/SpatialTransformer/_transform/_interpolate/mul_7_grad/tuple/control_dependency_1Ugradients/SpatialTransformer/_transform/_interpolate/sub_5_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
Ggradients/SpatialTransformer/_transform/_interpolate/sub_5_grad/ReshapeReshapeCgradients/SpatialTransformer/_transform/_interpolate/sub_5_grad/SumEgradients/SpatialTransformer/_transform/_interpolate/sub_5_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
Egradients/SpatialTransformer/_transform/_interpolate/sub_5_grad/Sum_1SumZgradients/SpatialTransformer/_transform/_interpolate/mul_7_grad/tuple/control_dependency_1Wgradients/SpatialTransformer/_transform/_interpolate/sub_5_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
Cgradients/SpatialTransformer/_transform/_interpolate/sub_5_grad/NegNegEgradients/SpatialTransformer/_transform/_interpolate/sub_5_grad/Sum_1*
T0*
_output_shapes
:
?
Igradients/SpatialTransformer/_transform/_interpolate/sub_5_grad/Reshape_1ReshapeCgradients/SpatialTransformer/_transform/_interpolate/sub_5_grad/NegGgradients/SpatialTransformer/_transform/_interpolate/sub_5_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:?????????
?
Pgradients/SpatialTransformer/_transform/_interpolate/sub_5_grad/tuple/group_depsNoOpH^gradients/SpatialTransformer/_transform/_interpolate/sub_5_grad/ReshapeJ^gradients/SpatialTransformer/_transform/_interpolate/sub_5_grad/Reshape_1
?
Xgradients/SpatialTransformer/_transform/_interpolate/sub_5_grad/tuple/control_dependencyIdentityGgradients/SpatialTransformer/_transform/_interpolate/sub_5_grad/ReshapeQ^gradients/SpatialTransformer/_transform/_interpolate/sub_5_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients/SpatialTransformer/_transform/_interpolate/sub_5_grad/Reshape*#
_output_shapes
:?????????
?
Zgradients/SpatialTransformer/_transform/_interpolate/sub_5_grad/tuple/control_dependency_1IdentityIgradients/SpatialTransformer/_transform/_interpolate/sub_5_grad/Reshape_1Q^gradients/SpatialTransformer/_transform/_interpolate/sub_5_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/SpatialTransformer/_transform/_interpolate/sub_5_grad/Reshape_1*#
_output_shapes
:?????????
?
Egradients/SpatialTransformer/_transform/_interpolate/sub_6_grad/ShapeShape2SpatialTransformer/_transform/_interpolate/truediv*
_output_shapes
:*
T0*
out_type0
?
Ggradients/SpatialTransformer/_transform/_interpolate/sub_6_grad/Shape_1Shape1SpatialTransformer/_transform/_interpolate/Cast_4*
T0*
out_type0*
_output_shapes
:
?
Ugradients/SpatialTransformer/_transform/_interpolate/sub_6_grad/BroadcastGradientArgsBroadcastGradientArgsEgradients/SpatialTransformer/_transform/_interpolate/sub_6_grad/ShapeGgradients/SpatialTransformer/_transform/_interpolate/sub_6_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
Cgradients/SpatialTransformer/_transform/_interpolate/sub_6_grad/SumSumXgradients/SpatialTransformer/_transform/_interpolate/mul_8_grad/tuple/control_dependencyUgradients/SpatialTransformer/_transform/_interpolate/sub_6_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
Ggradients/SpatialTransformer/_transform/_interpolate/sub_6_grad/ReshapeReshapeCgradients/SpatialTransformer/_transform/_interpolate/sub_6_grad/SumEgradients/SpatialTransformer/_transform/_interpolate/sub_6_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
Egradients/SpatialTransformer/_transform/_interpolate/sub_6_grad/Sum_1SumXgradients/SpatialTransformer/_transform/_interpolate/mul_8_grad/tuple/control_dependencyWgradients/SpatialTransformer/_transform/_interpolate/sub_6_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
Cgradients/SpatialTransformer/_transform/_interpolate/sub_6_grad/NegNegEgradients/SpatialTransformer/_transform/_interpolate/sub_6_grad/Sum_1*
T0*
_output_shapes
:
?
Igradients/SpatialTransformer/_transform/_interpolate/sub_6_grad/Reshape_1ReshapeCgradients/SpatialTransformer/_transform/_interpolate/sub_6_grad/NegGgradients/SpatialTransformer/_transform/_interpolate/sub_6_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:?????????
?
Pgradients/SpatialTransformer/_transform/_interpolate/sub_6_grad/tuple/group_depsNoOpH^gradients/SpatialTransformer/_transform/_interpolate/sub_6_grad/ReshapeJ^gradients/SpatialTransformer/_transform/_interpolate/sub_6_grad/Reshape_1
?
Xgradients/SpatialTransformer/_transform/_interpolate/sub_6_grad/tuple/control_dependencyIdentityGgradients/SpatialTransformer/_transform/_interpolate/sub_6_grad/ReshapeQ^gradients/SpatialTransformer/_transform/_interpolate/sub_6_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients/SpatialTransformer/_transform/_interpolate/sub_6_grad/Reshape*#
_output_shapes
:?????????
?
Zgradients/SpatialTransformer/_transform/_interpolate/sub_6_grad/tuple/control_dependency_1IdentityIgradients/SpatialTransformer/_transform/_interpolate/sub_6_grad/Reshape_1Q^gradients/SpatialTransformer/_transform/_interpolate/sub_6_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/SpatialTransformer/_transform/_interpolate/sub_6_grad/Reshape_1*#
_output_shapes
:?????????
?
Egradients/SpatialTransformer/_transform/_interpolate/sub_7_grad/ShapeShape1SpatialTransformer/_transform/_interpolate/Cast_7*
T0*
out_type0*
_output_shapes
:
?
Ggradients/SpatialTransformer/_transform/_interpolate/sub_7_grad/Shape_1Shape4SpatialTransformer/_transform/_interpolate/truediv_1*
T0*
out_type0*
_output_shapes
:
?
Ugradients/SpatialTransformer/_transform/_interpolate/sub_7_grad/BroadcastGradientArgsBroadcastGradientArgsEgradients/SpatialTransformer/_transform/_interpolate/sub_7_grad/ShapeGgradients/SpatialTransformer/_transform/_interpolate/sub_7_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
Cgradients/SpatialTransformer/_transform/_interpolate/sub_7_grad/SumSumZgradients/SpatialTransformer/_transform/_interpolate/mul_8_grad/tuple/control_dependency_1Ugradients/SpatialTransformer/_transform/_interpolate/sub_7_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
Ggradients/SpatialTransformer/_transform/_interpolate/sub_7_grad/ReshapeReshapeCgradients/SpatialTransformer/_transform/_interpolate/sub_7_grad/SumEgradients/SpatialTransformer/_transform/_interpolate/sub_7_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
Egradients/SpatialTransformer/_transform/_interpolate/sub_7_grad/Sum_1SumZgradients/SpatialTransformer/_transform/_interpolate/mul_8_grad/tuple/control_dependency_1Wgradients/SpatialTransformer/_transform/_interpolate/sub_7_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
?
Cgradients/SpatialTransformer/_transform/_interpolate/sub_7_grad/NegNegEgradients/SpatialTransformer/_transform/_interpolate/sub_7_grad/Sum_1*
_output_shapes
:*
T0
?
Igradients/SpatialTransformer/_transform/_interpolate/sub_7_grad/Reshape_1ReshapeCgradients/SpatialTransformer/_transform/_interpolate/sub_7_grad/NegGgradients/SpatialTransformer/_transform/_interpolate/sub_7_grad/Shape_1*#
_output_shapes
:?????????*
T0*
Tshape0
?
Pgradients/SpatialTransformer/_transform/_interpolate/sub_7_grad/tuple/group_depsNoOpH^gradients/SpatialTransformer/_transform/_interpolate/sub_7_grad/ReshapeJ^gradients/SpatialTransformer/_transform/_interpolate/sub_7_grad/Reshape_1
?
Xgradients/SpatialTransformer/_transform/_interpolate/sub_7_grad/tuple/control_dependencyIdentityGgradients/SpatialTransformer/_transform/_interpolate/sub_7_grad/ReshapeQ^gradients/SpatialTransformer/_transform/_interpolate/sub_7_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients/SpatialTransformer/_transform/_interpolate/sub_7_grad/Reshape*#
_output_shapes
:?????????
?
Zgradients/SpatialTransformer/_transform/_interpolate/sub_7_grad/tuple/control_dependency_1IdentityIgradients/SpatialTransformer/_transform/_interpolate/sub_7_grad/Reshape_1Q^gradients/SpatialTransformer/_transform/_interpolate/sub_7_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/SpatialTransformer/_transform/_interpolate/sub_7_grad/Reshape_1*#
_output_shapes
:?????????
?
Egradients/SpatialTransformer/_transform/_interpolate/sub_8_grad/ShapeShape2SpatialTransformer/_transform/_interpolate/truediv*
T0*
out_type0*
_output_shapes
:
?
Ggradients/SpatialTransformer/_transform/_interpolate/sub_8_grad/Shape_1Shape1SpatialTransformer/_transform/_interpolate/Cast_4*
T0*
out_type0*
_output_shapes
:
?
Ugradients/SpatialTransformer/_transform/_interpolate/sub_8_grad/BroadcastGradientArgsBroadcastGradientArgsEgradients/SpatialTransformer/_transform/_interpolate/sub_8_grad/ShapeGgradients/SpatialTransformer/_transform/_interpolate/sub_8_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
Cgradients/SpatialTransformer/_transform/_interpolate/sub_8_grad/SumSumXgradients/SpatialTransformer/_transform/_interpolate/mul_9_grad/tuple/control_dependencyUgradients/SpatialTransformer/_transform/_interpolate/sub_8_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
Ggradients/SpatialTransformer/_transform/_interpolate/sub_8_grad/ReshapeReshapeCgradients/SpatialTransformer/_transform/_interpolate/sub_8_grad/SumEgradients/SpatialTransformer/_transform/_interpolate/sub_8_grad/Shape*#
_output_shapes
:?????????*
T0*
Tshape0
?
Egradients/SpatialTransformer/_transform/_interpolate/sub_8_grad/Sum_1SumXgradients/SpatialTransformer/_transform/_interpolate/mul_9_grad/tuple/control_dependencyWgradients/SpatialTransformer/_transform/_interpolate/sub_8_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
Cgradients/SpatialTransformer/_transform/_interpolate/sub_8_grad/NegNegEgradients/SpatialTransformer/_transform/_interpolate/sub_8_grad/Sum_1*
T0*
_output_shapes
:
?
Igradients/SpatialTransformer/_transform/_interpolate/sub_8_grad/Reshape_1ReshapeCgradients/SpatialTransformer/_transform/_interpolate/sub_8_grad/NegGgradients/SpatialTransformer/_transform/_interpolate/sub_8_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:?????????
?
Pgradients/SpatialTransformer/_transform/_interpolate/sub_8_grad/tuple/group_depsNoOpH^gradients/SpatialTransformer/_transform/_interpolate/sub_8_grad/ReshapeJ^gradients/SpatialTransformer/_transform/_interpolate/sub_8_grad/Reshape_1
?
Xgradients/SpatialTransformer/_transform/_interpolate/sub_8_grad/tuple/control_dependencyIdentityGgradients/SpatialTransformer/_transform/_interpolate/sub_8_grad/ReshapeQ^gradients/SpatialTransformer/_transform/_interpolate/sub_8_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients/SpatialTransformer/_transform/_interpolate/sub_8_grad/Reshape*#
_output_shapes
:?????????
?
Zgradients/SpatialTransformer/_transform/_interpolate/sub_8_grad/tuple/control_dependency_1IdentityIgradients/SpatialTransformer/_transform/_interpolate/sub_8_grad/Reshape_1Q^gradients/SpatialTransformer/_transform/_interpolate/sub_8_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/SpatialTransformer/_transform/_interpolate/sub_8_grad/Reshape_1*#
_output_shapes
:?????????
?
Egradients/SpatialTransformer/_transform/_interpolate/sub_9_grad/ShapeShape4SpatialTransformer/_transform/_interpolate/truediv_1*
T0*
out_type0*
_output_shapes
:
?
Ggradients/SpatialTransformer/_transform/_interpolate/sub_9_grad/Shape_1Shape1SpatialTransformer/_transform/_interpolate/Cast_6*
T0*
out_type0*
_output_shapes
:
?
Ugradients/SpatialTransformer/_transform/_interpolate/sub_9_grad/BroadcastGradientArgsBroadcastGradientArgsEgradients/SpatialTransformer/_transform/_interpolate/sub_9_grad/ShapeGgradients/SpatialTransformer/_transform/_interpolate/sub_9_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
Cgradients/SpatialTransformer/_transform/_interpolate/sub_9_grad/SumSumZgradients/SpatialTransformer/_transform/_interpolate/mul_9_grad/tuple/control_dependency_1Ugradients/SpatialTransformer/_transform/_interpolate/sub_9_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
Ggradients/SpatialTransformer/_transform/_interpolate/sub_9_grad/ReshapeReshapeCgradients/SpatialTransformer/_transform/_interpolate/sub_9_grad/SumEgradients/SpatialTransformer/_transform/_interpolate/sub_9_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
Egradients/SpatialTransformer/_transform/_interpolate/sub_9_grad/Sum_1SumZgradients/SpatialTransformer/_transform/_interpolate/mul_9_grad/tuple/control_dependency_1Wgradients/SpatialTransformer/_transform/_interpolate/sub_9_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
Cgradients/SpatialTransformer/_transform/_interpolate/sub_9_grad/NegNegEgradients/SpatialTransformer/_transform/_interpolate/sub_9_grad/Sum_1*
T0*
_output_shapes
:
?
Igradients/SpatialTransformer/_transform/_interpolate/sub_9_grad/Reshape_1ReshapeCgradients/SpatialTransformer/_transform/_interpolate/sub_9_grad/NegGgradients/SpatialTransformer/_transform/_interpolate/sub_9_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:?????????
?
Pgradients/SpatialTransformer/_transform/_interpolate/sub_9_grad/tuple/group_depsNoOpH^gradients/SpatialTransformer/_transform/_interpolate/sub_9_grad/ReshapeJ^gradients/SpatialTransformer/_transform/_interpolate/sub_9_grad/Reshape_1
?
Xgradients/SpatialTransformer/_transform/_interpolate/sub_9_grad/tuple/control_dependencyIdentityGgradients/SpatialTransformer/_transform/_interpolate/sub_9_grad/ReshapeQ^gradients/SpatialTransformer/_transform/_interpolate/sub_9_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients/SpatialTransformer/_transform/_interpolate/sub_9_grad/Reshape*#
_output_shapes
:?????????
?
Zgradients/SpatialTransformer/_transform/_interpolate/sub_9_grad/tuple/control_dependency_1IdentityIgradients/SpatialTransformer/_transform/_interpolate/sub_9_grad/Reshape_1Q^gradients/SpatialTransformer/_transform/_interpolate/sub_9_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/SpatialTransformer/_transform/_interpolate/sub_9_grad/Reshape_1*#
_output_shapes
:?????????
?
gradients/AddNAddNZgradients/SpatialTransformer/_transform/_interpolate/sub_2_grad/tuple/control_dependency_1Zgradients/SpatialTransformer/_transform/_interpolate/sub_4_grad/tuple/control_dependency_1Xgradients/SpatialTransformer/_transform/_interpolate/sub_6_grad/tuple/control_dependencyXgradients/SpatialTransformer/_transform/_interpolate/sub_8_grad/tuple/control_dependency*
T0*\
_classR
PNloc:@gradients/SpatialTransformer/_transform/_interpolate/sub_2_grad/Reshape_1*
N*#
_output_shapes
:?????????
?
Ggradients/SpatialTransformer/_transform/_interpolate/truediv_grad/ShapeShape.SpatialTransformer/_transform/_interpolate/mul*
_output_shapes
:*
T0*
out_type0
?
Igradients/SpatialTransformer/_transform/_interpolate/truediv_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
Wgradients/SpatialTransformer/_transform/_interpolate/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsGgradients/SpatialTransformer/_transform/_interpolate/truediv_grad/ShapeIgradients/SpatialTransformer/_transform/_interpolate/truediv_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
Igradients/SpatialTransformer/_transform/_interpolate/truediv_grad/RealDivRealDivgradients/AddN4SpatialTransformer/_transform/_interpolate/truediv/y*
T0*#
_output_shapes
:?????????
?
Egradients/SpatialTransformer/_transform/_interpolate/truediv_grad/SumSumIgradients/SpatialTransformer/_transform/_interpolate/truediv_grad/RealDivWgradients/SpatialTransformer/_transform/_interpolate/truediv_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
?
Igradients/SpatialTransformer/_transform/_interpolate/truediv_grad/ReshapeReshapeEgradients/SpatialTransformer/_transform/_interpolate/truediv_grad/SumGgradients/SpatialTransformer/_transform/_interpolate/truediv_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
Egradients/SpatialTransformer/_transform/_interpolate/truediv_grad/NegNeg.SpatialTransformer/_transform/_interpolate/mul*#
_output_shapes
:?????????*
T0
?
Kgradients/SpatialTransformer/_transform/_interpolate/truediv_grad/RealDiv_1RealDivEgradients/SpatialTransformer/_transform/_interpolate/truediv_grad/Neg4SpatialTransformer/_transform/_interpolate/truediv/y*
T0*#
_output_shapes
:?????????
?
Kgradients/SpatialTransformer/_transform/_interpolate/truediv_grad/RealDiv_2RealDivKgradients/SpatialTransformer/_transform/_interpolate/truediv_grad/RealDiv_14SpatialTransformer/_transform/_interpolate/truediv/y*#
_output_shapes
:?????????*
T0
?
Egradients/SpatialTransformer/_transform/_interpolate/truediv_grad/mulMulgradients/AddNKgradients/SpatialTransformer/_transform/_interpolate/truediv_grad/RealDiv_2*#
_output_shapes
:?????????*
T0
?
Ggradients/SpatialTransformer/_transform/_interpolate/truediv_grad/Sum_1SumEgradients/SpatialTransformer/_transform/_interpolate/truediv_grad/mulYgradients/SpatialTransformer/_transform/_interpolate/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
Kgradients/SpatialTransformer/_transform/_interpolate/truediv_grad/Reshape_1ReshapeGgradients/SpatialTransformer/_transform/_interpolate/truediv_grad/Sum_1Igradients/SpatialTransformer/_transform/_interpolate/truediv_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
Rgradients/SpatialTransformer/_transform/_interpolate/truediv_grad/tuple/group_depsNoOpJ^gradients/SpatialTransformer/_transform/_interpolate/truediv_grad/ReshapeL^gradients/SpatialTransformer/_transform/_interpolate/truediv_grad/Reshape_1
?
Zgradients/SpatialTransformer/_transform/_interpolate/truediv_grad/tuple/control_dependencyIdentityIgradients/SpatialTransformer/_transform/_interpolate/truediv_grad/ReshapeS^gradients/SpatialTransformer/_transform/_interpolate/truediv_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/SpatialTransformer/_transform/_interpolate/truediv_grad/Reshape*#
_output_shapes
:?????????
?
\gradients/SpatialTransformer/_transform/_interpolate/truediv_grad/tuple/control_dependency_1IdentityKgradients/SpatialTransformer/_transform/_interpolate/truediv_grad/Reshape_1S^gradients/SpatialTransformer/_transform/_interpolate/truediv_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/SpatialTransformer/_transform/_interpolate/truediv_grad/Reshape_1*
_output_shapes
: 
?
gradients/AddN_1AddNZgradients/SpatialTransformer/_transform/_interpolate/sub_3_grad/tuple/control_dependency_1Xgradients/SpatialTransformer/_transform/_interpolate/sub_5_grad/tuple/control_dependencyZgradients/SpatialTransformer/_transform/_interpolate/sub_7_grad/tuple/control_dependency_1Xgradients/SpatialTransformer/_transform/_interpolate/sub_9_grad/tuple/control_dependency*
N*#
_output_shapes
:?????????*
T0*\
_classR
PNloc:@gradients/SpatialTransformer/_transform/_interpolate/sub_3_grad/Reshape_1
?
Igradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/ShapeShape0SpatialTransformer/_transform/_interpolate/mul_1*
T0*
out_type0*
_output_shapes
:
?
Kgradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
Ygradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/BroadcastGradientArgsBroadcastGradientArgsIgradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/ShapeKgradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
Kgradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/RealDivRealDivgradients/AddN_16SpatialTransformer/_transform/_interpolate/truediv_1/y*
T0*#
_output_shapes
:?????????
?
Ggradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/SumSumKgradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/RealDivYgradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
Kgradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/ReshapeReshapeGgradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/SumIgradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/Shape*
Tshape0*#
_output_shapes
:?????????*
T0
?
Ggradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/NegNeg0SpatialTransformer/_transform/_interpolate/mul_1*
T0*#
_output_shapes
:?????????
?
Mgradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/RealDiv_1RealDivGgradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/Neg6SpatialTransformer/_transform/_interpolate/truediv_1/y*#
_output_shapes
:?????????*
T0
?
Mgradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/RealDiv_2RealDivMgradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/RealDiv_16SpatialTransformer/_transform/_interpolate/truediv_1/y*
T0*#
_output_shapes
:?????????
?
Ggradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/mulMulgradients/AddN_1Mgradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/RealDiv_2*#
_output_shapes
:?????????*
T0
?
Igradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/Sum_1SumGgradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/mul[gradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
Mgradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/Reshape_1ReshapeIgradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/Sum_1Kgradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
Tgradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/tuple/group_depsNoOpL^gradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/ReshapeN^gradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/Reshape_1
?
\gradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/tuple/control_dependencyIdentityKgradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/ReshapeU^gradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/Reshape*#
_output_shapes
:?????????
?
^gradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/tuple/control_dependency_1IdentityMgradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/Reshape_1U^gradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/Reshape_1*
_output_shapes
: 
?
Cgradients/SpatialTransformer/_transform/_interpolate/mul_grad/ShapeShape.SpatialTransformer/_transform/_interpolate/add*
_output_shapes
:*
T0*
out_type0
?
Egradients/SpatialTransformer/_transform/_interpolate/mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
Sgradients/SpatialTransformer/_transform/_interpolate/mul_grad/BroadcastGradientArgsBroadcastGradientArgsCgradients/SpatialTransformer/_transform/_interpolate/mul_grad/ShapeEgradients/SpatialTransformer/_transform/_interpolate/mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
Agradients/SpatialTransformer/_transform/_interpolate/mul_grad/MulMulZgradients/SpatialTransformer/_transform/_interpolate/truediv_grad/tuple/control_dependency1SpatialTransformer/_transform/_interpolate/Cast_1*
T0*#
_output_shapes
:?????????
?
Agradients/SpatialTransformer/_transform/_interpolate/mul_grad/SumSumAgradients/SpatialTransformer/_transform/_interpolate/mul_grad/MulSgradients/SpatialTransformer/_transform/_interpolate/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
Egradients/SpatialTransformer/_transform/_interpolate/mul_grad/ReshapeReshapeAgradients/SpatialTransformer/_transform/_interpolate/mul_grad/SumCgradients/SpatialTransformer/_transform/_interpolate/mul_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
Cgradients/SpatialTransformer/_transform/_interpolate/mul_grad/Mul_1Mul.SpatialTransformer/_transform/_interpolate/addZgradients/SpatialTransformer/_transform/_interpolate/truediv_grad/tuple/control_dependency*#
_output_shapes
:?????????*
T0
?
Cgradients/SpatialTransformer/_transform/_interpolate/mul_grad/Sum_1SumCgradients/SpatialTransformer/_transform/_interpolate/mul_grad/Mul_1Ugradients/SpatialTransformer/_transform/_interpolate/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
Ggradients/SpatialTransformer/_transform/_interpolate/mul_grad/Reshape_1ReshapeCgradients/SpatialTransformer/_transform/_interpolate/mul_grad/Sum_1Egradients/SpatialTransformer/_transform/_interpolate/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
Ngradients/SpatialTransformer/_transform/_interpolate/mul_grad/tuple/group_depsNoOpF^gradients/SpatialTransformer/_transform/_interpolate/mul_grad/ReshapeH^gradients/SpatialTransformer/_transform/_interpolate/mul_grad/Reshape_1
?
Vgradients/SpatialTransformer/_transform/_interpolate/mul_grad/tuple/control_dependencyIdentityEgradients/SpatialTransformer/_transform/_interpolate/mul_grad/ReshapeO^gradients/SpatialTransformer/_transform/_interpolate/mul_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients/SpatialTransformer/_transform/_interpolate/mul_grad/Reshape*#
_output_shapes
:?????????
?
Xgradients/SpatialTransformer/_transform/_interpolate/mul_grad/tuple/control_dependency_1IdentityGgradients/SpatialTransformer/_transform/_interpolate/mul_grad/Reshape_1O^gradients/SpatialTransformer/_transform/_interpolate/mul_grad/tuple/group_deps*Z
_classP
NLloc:@gradients/SpatialTransformer/_transform/_interpolate/mul_grad/Reshape_1*
_output_shapes
: *
T0
?
Egradients/SpatialTransformer/_transform/_interpolate/mul_1_grad/ShapeShape0SpatialTransformer/_transform/_interpolate/add_1*
T0*
out_type0*
_output_shapes
:
?
Ggradients/SpatialTransformer/_transform/_interpolate/mul_1_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
Ugradients/SpatialTransformer/_transform/_interpolate/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsEgradients/SpatialTransformer/_transform/_interpolate/mul_1_grad/ShapeGgradients/SpatialTransformer/_transform/_interpolate/mul_1_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
Cgradients/SpatialTransformer/_transform/_interpolate/mul_1_grad/MulMul\gradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/tuple/control_dependency/SpatialTransformer/_transform/_interpolate/Cast*#
_output_shapes
:?????????*
T0
?
Cgradients/SpatialTransformer/_transform/_interpolate/mul_1_grad/SumSumCgradients/SpatialTransformer/_transform/_interpolate/mul_1_grad/MulUgradients/SpatialTransformer/_transform/_interpolate/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
Ggradients/SpatialTransformer/_transform/_interpolate/mul_1_grad/ReshapeReshapeCgradients/SpatialTransformer/_transform/_interpolate/mul_1_grad/SumEgradients/SpatialTransformer/_transform/_interpolate/mul_1_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
Egradients/SpatialTransformer/_transform/_interpolate/mul_1_grad/Mul_1Mul0SpatialTransformer/_transform/_interpolate/add_1\gradients/SpatialTransformer/_transform/_interpolate/truediv_1_grad/tuple/control_dependency*
T0*#
_output_shapes
:?????????
?
Egradients/SpatialTransformer/_transform/_interpolate/mul_1_grad/Sum_1SumEgradients/SpatialTransformer/_transform/_interpolate/mul_1_grad/Mul_1Wgradients/SpatialTransformer/_transform/_interpolate/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
Igradients/SpatialTransformer/_transform/_interpolate/mul_1_grad/Reshape_1ReshapeEgradients/SpatialTransformer/_transform/_interpolate/mul_1_grad/Sum_1Ggradients/SpatialTransformer/_transform/_interpolate/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
Pgradients/SpatialTransformer/_transform/_interpolate/mul_1_grad/tuple/group_depsNoOpH^gradients/SpatialTransformer/_transform/_interpolate/mul_1_grad/ReshapeJ^gradients/SpatialTransformer/_transform/_interpolate/mul_1_grad/Reshape_1
?
Xgradients/SpatialTransformer/_transform/_interpolate/mul_1_grad/tuple/control_dependencyIdentityGgradients/SpatialTransformer/_transform/_interpolate/mul_1_grad/ReshapeQ^gradients/SpatialTransformer/_transform/_interpolate/mul_1_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients/SpatialTransformer/_transform/_interpolate/mul_1_grad/Reshape*#
_output_shapes
:?????????
?
Zgradients/SpatialTransformer/_transform/_interpolate/mul_1_grad/tuple/control_dependency_1IdentityIgradients/SpatialTransformer/_transform/_interpolate/mul_1_grad/Reshape_1Q^gradients/SpatialTransformer/_transform/_interpolate/mul_1_grad/tuple/group_deps*
_output_shapes
: *
T0*\
_classR
PNloc:@gradients/SpatialTransformer/_transform/_interpolate/mul_1_grad/Reshape_1
?
Cgradients/SpatialTransformer/_transform/_interpolate/add_grad/ShapeShape'SpatialTransformer/_transform/Reshape_3*
T0*
out_type0*
_output_shapes
:
?
Egradients/SpatialTransformer/_transform/_interpolate/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
Sgradients/SpatialTransformer/_transform/_interpolate/add_grad/BroadcastGradientArgsBroadcastGradientArgsCgradients/SpatialTransformer/_transform/_interpolate/add_grad/ShapeEgradients/SpatialTransformer/_transform/_interpolate/add_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
Agradients/SpatialTransformer/_transform/_interpolate/add_grad/SumSumVgradients/SpatialTransformer/_transform/_interpolate/mul_grad/tuple/control_dependencySgradients/SpatialTransformer/_transform/_interpolate/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
Egradients/SpatialTransformer/_transform/_interpolate/add_grad/ReshapeReshapeAgradients/SpatialTransformer/_transform/_interpolate/add_grad/SumCgradients/SpatialTransformer/_transform/_interpolate/add_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
Cgradients/SpatialTransformer/_transform/_interpolate/add_grad/Sum_1SumVgradients/SpatialTransformer/_transform/_interpolate/mul_grad/tuple/control_dependencyUgradients/SpatialTransformer/_transform/_interpolate/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
?
Ggradients/SpatialTransformer/_transform/_interpolate/add_grad/Reshape_1ReshapeCgradients/SpatialTransformer/_transform/_interpolate/add_grad/Sum_1Egradients/SpatialTransformer/_transform/_interpolate/add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
?
Ngradients/SpatialTransformer/_transform/_interpolate/add_grad/tuple/group_depsNoOpF^gradients/SpatialTransformer/_transform/_interpolate/add_grad/ReshapeH^gradients/SpatialTransformer/_transform/_interpolate/add_grad/Reshape_1
?
Vgradients/SpatialTransformer/_transform/_interpolate/add_grad/tuple/control_dependencyIdentityEgradients/SpatialTransformer/_transform/_interpolate/add_grad/ReshapeO^gradients/SpatialTransformer/_transform/_interpolate/add_grad/tuple/group_deps*#
_output_shapes
:?????????*
T0*X
_classN
LJloc:@gradients/SpatialTransformer/_transform/_interpolate/add_grad/Reshape
?
Xgradients/SpatialTransformer/_transform/_interpolate/add_grad/tuple/control_dependency_1IdentityGgradients/SpatialTransformer/_transform/_interpolate/add_grad/Reshape_1O^gradients/SpatialTransformer/_transform/_interpolate/add_grad/tuple/group_deps*
_output_shapes
: *
T0*Z
_classP
NLloc:@gradients/SpatialTransformer/_transform/_interpolate/add_grad/Reshape_1
?
Egradients/SpatialTransformer/_transform/_interpolate/add_1_grad/ShapeShape'SpatialTransformer/_transform/Reshape_4*
T0*
out_type0*
_output_shapes
:
?
Ggradients/SpatialTransformer/_transform/_interpolate/add_1_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
Ugradients/SpatialTransformer/_transform/_interpolate/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsEgradients/SpatialTransformer/_transform/_interpolate/add_1_grad/ShapeGgradients/SpatialTransformer/_transform/_interpolate/add_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
Cgradients/SpatialTransformer/_transform/_interpolate/add_1_grad/SumSumXgradients/SpatialTransformer/_transform/_interpolate/mul_1_grad/tuple/control_dependencyUgradients/SpatialTransformer/_transform/_interpolate/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
Ggradients/SpatialTransformer/_transform/_interpolate/add_1_grad/ReshapeReshapeCgradients/SpatialTransformer/_transform/_interpolate/add_1_grad/SumEgradients/SpatialTransformer/_transform/_interpolate/add_1_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
Egradients/SpatialTransformer/_transform/_interpolate/add_1_grad/Sum_1SumXgradients/SpatialTransformer/_transform/_interpolate/mul_1_grad/tuple/control_dependencyWgradients/SpatialTransformer/_transform/_interpolate/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
Igradients/SpatialTransformer/_transform/_interpolate/add_1_grad/Reshape_1ReshapeEgradients/SpatialTransformer/_transform/_interpolate/add_1_grad/Sum_1Ggradients/SpatialTransformer/_transform/_interpolate/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
Pgradients/SpatialTransformer/_transform/_interpolate/add_1_grad/tuple/group_depsNoOpH^gradients/SpatialTransformer/_transform/_interpolate/add_1_grad/ReshapeJ^gradients/SpatialTransformer/_transform/_interpolate/add_1_grad/Reshape_1
?
Xgradients/SpatialTransformer/_transform/_interpolate/add_1_grad/tuple/control_dependencyIdentityGgradients/SpatialTransformer/_transform/_interpolate/add_1_grad/ReshapeQ^gradients/SpatialTransformer/_transform/_interpolate/add_1_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients/SpatialTransformer/_transform/_interpolate/add_1_grad/Reshape*#
_output_shapes
:?????????
?
Zgradients/SpatialTransformer/_transform/_interpolate/add_1_grad/tuple/control_dependency_1IdentityIgradients/SpatialTransformer/_transform/_interpolate/add_1_grad/Reshape_1Q^gradients/SpatialTransformer/_transform/_interpolate/add_1_grad/tuple/group_deps*\
_classR
PNloc:@gradients/SpatialTransformer/_transform/_interpolate/add_1_grad/Reshape_1*
_output_shapes
: *
T0
?
<gradients/SpatialTransformer/_transform/Reshape_3_grad/ShapeShape#SpatialTransformer/_transform/Slice*
out_type0*
_output_shapes
:*
T0
?
>gradients/SpatialTransformer/_transform/Reshape_3_grad/ReshapeReshapeVgradients/SpatialTransformer/_transform/_interpolate/add_grad/tuple/control_dependency<gradients/SpatialTransformer/_transform/Reshape_3_grad/Shape*
T0*
Tshape0*4
_output_shapes"
 :??????????????????
?
<gradients/SpatialTransformer/_transform/Reshape_4_grad/ShapeShape%SpatialTransformer/_transform/Slice_1*
_output_shapes
:*
T0*
out_type0
?
>gradients/SpatialTransformer/_transform/Reshape_4_grad/ReshapeReshapeXgradients/SpatialTransformer/_transform/_interpolate/add_1_grad/tuple/control_dependency<gradients/SpatialTransformer/_transform/Reshape_4_grad/Shape*
T0*
Tshape0*4
_output_shapes"
 :??????????????????
y
7gradients/SpatialTransformer/_transform/Slice_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
?
8gradients/SpatialTransformer/_transform/Slice_grad/ShapeShape#SpatialTransformer/_transform/Slice*
_output_shapes
:*
T0*
out_type0
|
:gradients/SpatialTransformer/_transform/Slice_grad/stack/1Const*
dtype0*
_output_shapes
: *
value	B :
?
8gradients/SpatialTransformer/_transform/Slice_grad/stackPack7gradients/SpatialTransformer/_transform/Slice_grad/Rank:gradients/SpatialTransformer/_transform/Slice_grad/stack/1*
T0*

axis *
N*
_output_shapes
:
?
:gradients/SpatialTransformer/_transform/Slice_grad/ReshapeReshape)SpatialTransformer/_transform/Slice/begin8gradients/SpatialTransformer/_transform/Slice_grad/stack*
T0*
Tshape0*
_output_shapes

:
?
:gradients/SpatialTransformer/_transform/Slice_grad/Shape_1Shape$SpatialTransformer/_transform/MatMul*
T0*
out_type0*
_output_shapes
:
?
6gradients/SpatialTransformer/_transform/Slice_grad/subSub:gradients/SpatialTransformer/_transform/Slice_grad/Shape_18gradients/SpatialTransformer/_transform/Slice_grad/Shape*
T0*
_output_shapes
:
?
8gradients/SpatialTransformer/_transform/Slice_grad/sub_1Sub6gradients/SpatialTransformer/_transform/Slice_grad/sub)SpatialTransformer/_transform/Slice/begin*
T0*
_output_shapes
:
?
<gradients/SpatialTransformer/_transform/Slice_grad/Reshape_1Reshape8gradients/SpatialTransformer/_transform/Slice_grad/sub_18gradients/SpatialTransformer/_transform/Slice_grad/stack*
T0*
Tshape0*
_output_shapes

:
?
>gradients/SpatialTransformer/_transform/Slice_grad/concat/axisConst*
_output_shapes
: *
value	B :*
dtype0
?
9gradients/SpatialTransformer/_transform/Slice_grad/concatConcatV2:gradients/SpatialTransformer/_transform/Slice_grad/Reshape<gradients/SpatialTransformer/_transform/Slice_grad/Reshape_1>gradients/SpatialTransformer/_transform/Slice_grad/concat/axis*
N*
_output_shapes

:*

Tidx0*
T0
?
6gradients/SpatialTransformer/_transform/Slice_grad/PadPad>gradients/SpatialTransformer/_transform/Reshape_3_grad/Reshape9gradients/SpatialTransformer/_transform/Slice_grad/concat*
	Tpaddings0*4
_output_shapes"
 :??????????????????*
T0
{
9gradients/SpatialTransformer/_transform/Slice_1_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
?
:gradients/SpatialTransformer/_transform/Slice_1_grad/ShapeShape%SpatialTransformer/_transform/Slice_1*
T0*
out_type0*
_output_shapes
:
~
<gradients/SpatialTransformer/_transform/Slice_1_grad/stack/1Const*
dtype0*
_output_shapes
: *
value	B :
?
:gradients/SpatialTransformer/_transform/Slice_1_grad/stackPack9gradients/SpatialTransformer/_transform/Slice_1_grad/Rank<gradients/SpatialTransformer/_transform/Slice_1_grad/stack/1*
T0*

axis *
N*
_output_shapes
:
?
<gradients/SpatialTransformer/_transform/Slice_1_grad/ReshapeReshape+SpatialTransformer/_transform/Slice_1/begin:gradients/SpatialTransformer/_transform/Slice_1_grad/stack*
T0*
Tshape0*
_output_shapes

:
?
<gradients/SpatialTransformer/_transform/Slice_1_grad/Shape_1Shape$SpatialTransformer/_transform/MatMul*
T0*
out_type0*
_output_shapes
:
?
8gradients/SpatialTransformer/_transform/Slice_1_grad/subSub<gradients/SpatialTransformer/_transform/Slice_1_grad/Shape_1:gradients/SpatialTransformer/_transform/Slice_1_grad/Shape*
T0*
_output_shapes
:
?
:gradients/SpatialTransformer/_transform/Slice_1_grad/sub_1Sub8gradients/SpatialTransformer/_transform/Slice_1_grad/sub+SpatialTransformer/_transform/Slice_1/begin*
_output_shapes
:*
T0
?
>gradients/SpatialTransformer/_transform/Slice_1_grad/Reshape_1Reshape:gradients/SpatialTransformer/_transform/Slice_1_grad/sub_1:gradients/SpatialTransformer/_transform/Slice_1_grad/stack*
_output_shapes

:*
T0*
Tshape0
?
@gradients/SpatialTransformer/_transform/Slice_1_grad/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
?
;gradients/SpatialTransformer/_transform/Slice_1_grad/concatConcatV2<gradients/SpatialTransformer/_transform/Slice_1_grad/Reshape>gradients/SpatialTransformer/_transform/Slice_1_grad/Reshape_1@gradients/SpatialTransformer/_transform/Slice_1_grad/concat/axis*

Tidx0*
T0*
N*
_output_shapes

:
?
8gradients/SpatialTransformer/_transform/Slice_1_grad/PadPad>gradients/SpatialTransformer/_transform/Reshape_4_grad/Reshape;gradients/SpatialTransformer/_transform/Slice_1_grad/concat*
T0*
	Tpaddings0*4
_output_shapes"
 :??????????????????
?
gradients/AddN_2AddN6gradients/SpatialTransformer/_transform/Slice_grad/Pad8gradients/SpatialTransformer/_transform/Slice_1_grad/Pad*
T0*I
_class?
=;loc:@gradients/SpatialTransformer/_transform/Slice_grad/Pad*
N*4
_output_shapes"
 :??????????????????
?
:gradients/SpatialTransformer/_transform/MatMul_grad/MatMulBatchMatMulV2gradients/AddN_2'SpatialTransformer/_transform/Reshape_2*
adj_x( *
adj_y(*
T0*+
_output_shapes
:?????????
?
<gradients/SpatialTransformer/_transform/MatMul_grad/MatMul_1BatchMatMulV2%SpatialTransformer/_transform/Reshapegradients/AddN_2*
adj_y( *
T0*4
_output_shapes"
 :??????????????????*
adj_x(
?
9gradients/SpatialTransformer/_transform/MatMul_grad/ShapeShape%SpatialTransformer/_transform/Reshape*
T0*
out_type0*
_output_shapes
:
?
;gradients/SpatialTransformer/_transform/MatMul_grad/Shape_1Shape'SpatialTransformer/_transform/Reshape_2*
_output_shapes
:*
T0*
out_type0
?
Ggradients/SpatialTransformer/_transform/MatMul_grad/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
?
Igradients/SpatialTransformer/_transform/MatMul_grad/strided_slice/stack_1Const*
_output_shapes
:*
valueB:
?????????*
dtype0
?
Igradients/SpatialTransformer/_transform/MatMul_grad/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
Agradients/SpatialTransformer/_transform/MatMul_grad/strided_sliceStridedSlice9gradients/SpatialTransformer/_transform/MatMul_grad/ShapeGgradients/SpatialTransformer/_transform/MatMul_grad/strided_slice/stackIgradients/SpatialTransformer/_transform/MatMul_grad/strided_slice/stack_1Igradients/SpatialTransformer/_transform/MatMul_grad/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:
?
Igradients/SpatialTransformer/_transform/MatMul_grad/strided_slice_1/stackConst*
_output_shapes
:*
valueB: *
dtype0
?
Kgradients/SpatialTransformer/_transform/MatMul_grad/strided_slice_1/stack_1Const*
valueB:
?????????*
dtype0*
_output_shapes
:
?
Kgradients/SpatialTransformer/_transform/MatMul_grad/strided_slice_1/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
?
Cgradients/SpatialTransformer/_transform/MatMul_grad/strided_slice_1StridedSlice;gradients/SpatialTransformer/_transform/MatMul_grad/Shape_1Igradients/SpatialTransformer/_transform/MatMul_grad/strided_slice_1/stackKgradients/SpatialTransformer/_transform/MatMul_grad/strided_slice_1/stack_1Kgradients/SpatialTransformer/_transform/MatMul_grad/strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:
?
Igradients/SpatialTransformer/_transform/MatMul_grad/BroadcastGradientArgsBroadcastGradientArgsAgradients/SpatialTransformer/_transform/MatMul_grad/strided_sliceCgradients/SpatialTransformer/_transform/MatMul_grad/strided_slice_1*2
_output_shapes 
:?????????:?????????*
T0
?
7gradients/SpatialTransformer/_transform/MatMul_grad/SumSum:gradients/SpatialTransformer/_transform/MatMul_grad/MatMulIgradients/SpatialTransformer/_transform/MatMul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
;gradients/SpatialTransformer/_transform/MatMul_grad/ReshapeReshape7gradients/SpatialTransformer/_transform/MatMul_grad/Sum9gradients/SpatialTransformer/_transform/MatMul_grad/Shape*
T0*
Tshape0*+
_output_shapes
:?????????
?
9gradients/SpatialTransformer/_transform/MatMul_grad/Sum_1Sum<gradients/SpatialTransformer/_transform/MatMul_grad/MatMul_1Kgradients/SpatialTransformer/_transform/MatMul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
=gradients/SpatialTransformer/_transform/MatMul_grad/Reshape_1Reshape9gradients/SpatialTransformer/_transform/MatMul_grad/Sum_1;gradients/SpatialTransformer/_transform/MatMul_grad/Shape_1*4
_output_shapes"
 :??????????????????*
T0*
Tshape0
?
Dgradients/SpatialTransformer/_transform/MatMul_grad/tuple/group_depsNoOp<^gradients/SpatialTransformer/_transform/MatMul_grad/Reshape>^gradients/SpatialTransformer/_transform/MatMul_grad/Reshape_1
?
Lgradients/SpatialTransformer/_transform/MatMul_grad/tuple/control_dependencyIdentity;gradients/SpatialTransformer/_transform/MatMul_grad/ReshapeE^gradients/SpatialTransformer/_transform/MatMul_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/SpatialTransformer/_transform/MatMul_grad/Reshape*+
_output_shapes
:?????????
?
Ngradients/SpatialTransformer/_transform/MatMul_grad/tuple/control_dependency_1Identity=gradients/SpatialTransformer/_transform/MatMul_grad/Reshape_1E^gradients/SpatialTransformer/_transform/MatMul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/SpatialTransformer/_transform/MatMul_grad/Reshape_1*4
_output_shapes"
 :??????????????????
?
:gradients/SpatialTransformer/_transform/Reshape_grad/ShapeShapeTanh_1*
T0*
out_type0*
_output_shapes
:
?
<gradients/SpatialTransformer/_transform/Reshape_grad/ReshapeReshapeLgradients/SpatialTransformer/_transform/MatMul_grad/tuple/control_dependency:gradients/SpatialTransformer/_transform/Reshape_grad/Shape*
Tshape0*'
_output_shapes
:?????????*
T0
?
gradients/Tanh_1_grad/TanhGradTanhGradTanh_1<gradients/SpatialTransformer/_transform/Reshape_grad/Reshape*
T0*'
_output_shapes
:?????????
b
gradients/add_1_grad/ShapeShapeMatMul_1*
T0*
out_type0*
_output_shapes
:
f
gradients/add_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
?
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients/add_1_grad/SumSumgradients/Tanh_1_grad/TanhGrad*gradients/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
Tshape0*'
_output_shapes
:?????????*
T0
?
gradients/add_1_grad/Sum_1Sumgradients/Tanh_1_grad/TanhGrad,gradients/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
?
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_1_grad/Reshape*'
_output_shapes
:?????????*
T0
?
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1*
_output_shapes
:
?
gradients/MatMul_1_grad/MatMulMatMul-gradients/add_1_grad/tuple/control_dependencyVariable_2/read*
T0*'
_output_shapes
:?????????*
transpose_a( *
transpose_b(
?
 gradients/MatMul_1_grad/MatMul_1MatMuldropout/mul_1-gradients/add_1_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
?
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul*'
_output_shapes
:?????????
?
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
_output_shapes

:
v
"gradients/dropout/mul_1_grad/ShapeShapedropout/mul*
T0*
out_type0*#
_output_shapes
:?????????
y
$gradients/dropout/mul_1_grad/Shape_1Shapedropout/Cast*
T0*
out_type0*#
_output_shapes
:?????????
?
2gradients/dropout/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/dropout/mul_1_grad/Shape$gradients/dropout/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
 gradients/dropout/mul_1_grad/MulMul0gradients/MatMul_1_grad/tuple/control_dependencydropout/Cast*
T0*
_output_shapes
:
?
 gradients/dropout/mul_1_grad/SumSum gradients/dropout/mul_1_grad/Mul2gradients/dropout/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
?
$gradients/dropout/mul_1_grad/ReshapeReshape gradients/dropout/mul_1_grad/Sum"gradients/dropout/mul_1_grad/Shape*
T0*
Tshape0*
_output_shapes
:
?
"gradients/dropout/mul_1_grad/Mul_1Muldropout/mul0gradients/MatMul_1_grad/tuple/control_dependency*
T0*
_output_shapes
:
?
"gradients/dropout/mul_1_grad/Sum_1Sum"gradients/dropout/mul_1_grad/Mul_14gradients/dropout/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
&gradients/dropout/mul_1_grad/Reshape_1Reshape"gradients/dropout/mul_1_grad/Sum_1$gradients/dropout/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
?
-gradients/dropout/mul_1_grad/tuple/group_depsNoOp%^gradients/dropout/mul_1_grad/Reshape'^gradients/dropout/mul_1_grad/Reshape_1
?
5gradients/dropout/mul_1_grad/tuple/control_dependencyIdentity$gradients/dropout/mul_1_grad/Reshape.^gradients/dropout/mul_1_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dropout/mul_1_grad/Reshape*
_output_shapes
:
?
7gradients/dropout/mul_1_grad/tuple/control_dependency_1Identity&gradients/dropout/mul_1_grad/Reshape_1.^gradients/dropout/mul_1_grad/tuple/group_deps*
_output_shapes
:*
T0*9
_class/
-+loc:@gradients/dropout/mul_1_grad/Reshape_1
d
 gradients/dropout/mul_grad/ShapeShapeTanh*
T0*
out_type0*
_output_shapes
:
z
"gradients/dropout/mul_grad/Shape_1Shapedropout/truediv*
T0*
out_type0*#
_output_shapes
:?????????
?
0gradients/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/mul_grad/Shape"gradients/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients/dropout/mul_grad/MulMul5gradients/dropout/mul_1_grad/tuple/control_dependencydropout/truediv*
_output_shapes
:*
T0
?
gradients/dropout/mul_grad/SumSumgradients/dropout/mul_grad/Mul0gradients/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
"gradients/dropout/mul_grad/ReshapeReshapegradients/dropout/mul_grad/Sum gradients/dropout/mul_grad/Shape*'
_output_shapes
:?????????*
T0*
Tshape0
?
 gradients/dropout/mul_grad/Mul_1MulTanh5gradients/dropout/mul_1_grad/tuple/control_dependency*
T0*
_output_shapes
:
?
 gradients/dropout/mul_grad/Sum_1Sum gradients/dropout/mul_grad/Mul_12gradients/dropout/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
$gradients/dropout/mul_grad/Reshape_1Reshape gradients/dropout/mul_grad/Sum_1"gradients/dropout/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

+gradients/dropout/mul_grad/tuple/group_depsNoOp#^gradients/dropout/mul_grad/Reshape%^gradients/dropout/mul_grad/Reshape_1
?
3gradients/dropout/mul_grad/tuple/control_dependencyIdentity"gradients/dropout/mul_grad/Reshape,^gradients/dropout/mul_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/dropout/mul_grad/Reshape*'
_output_shapes
:?????????
?
5gradients/dropout/mul_grad/tuple/control_dependency_1Identity$gradients/dropout/mul_grad/Reshape_1,^gradients/dropout/mul_grad/tuple/group_deps*
_output_shapes
:*
T0*7
_class-
+)loc:@gradients/dropout/mul_grad/Reshape_1
?
gradients/Tanh_grad/TanhGradTanhGradTanh3gradients/dropout/mul_grad/tuple/control_dependency*'
_output_shapes
:?????????*
T0
^
gradients/add_grad/ShapeShapeMatMul*
_output_shapes
:*
T0*
out_type0
d
gradients/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
?
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients/add_grad/SumSumgradients/Tanh_grad/TanhGrad(gradients/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
Tshape0*'
_output_shapes
:?????????*
T0
?
gradients/add_grad/Sum_1Sumgradients/Tanh_grad/TanhGrad*gradients/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
?
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
?
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape*'
_output_shapes
:?????????
?
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
_output_shapes
:*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1
?
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyVariable/read*(
_output_shapes
:??????????*
transpose_a( *
transpose_b(*
T0
?
gradients/MatMul_grad/MatMul_1MatMulx+gradients/add_grad/tuple/control_dependency*
_output_shapes
:	?*
transpose_a(*
transpose_b( *
T0
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
?
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*(
_output_shapes
:??????????
?
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
_output_shapes
:	?
{
beta1_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *fff?*
_class
loc:@Variable
?
beta1_power
VariableV2*
_output_shapes
: *
shared_name *
_class
loc:@Variable*
	container *
shape: *
dtype0
?
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Variable
g
beta1_power/readIdentitybeta1_power*
_output_shapes
: *
T0*
_class
loc:@Variable
{
beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *w??*
_class
loc:@Variable
?
beta2_power
VariableV2*
_class
loc:@Variable*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
?
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
g
beta2_power/readIdentitybeta2_power*
T0*
_class
loc:@Variable*
_output_shapes
: 
?
/Variable/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Variable*
valueB"     *
dtype0*
_output_shapes
:
?
%Variable/Adam/Initializer/zeros/ConstConst*
_class
loc:@Variable*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Variable/Adam/Initializer/zerosFill/Variable/Adam/Initializer/zeros/shape_as_tensor%Variable/Adam/Initializer/zeros/Const*
T0*
_class
loc:@Variable*

index_type0*
_output_shapes
:	?
?
Variable/Adam
VariableV2*
_class
loc:@Variable*
	container *
shape:	?*
dtype0*
_output_shapes
:	?*
shared_name 
?
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:	?*
use_locking(*
T0*
_class
loc:@Variable
t
Variable/Adam/readIdentityVariable/Adam*
T0*
_class
loc:@Variable*
_output_shapes
:	?
?
1Variable/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
_class
loc:@Variable*
valueB"     
?
'Variable/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *
_class
loc:@Variable*
valueB
 *    *
dtype0
?
!Variable/Adam_1/Initializer/zerosFill1Variable/Adam_1/Initializer/zeros/shape_as_tensor'Variable/Adam_1/Initializer/zeros/Const*
_output_shapes
:	?*
T0*
_class
loc:@Variable*

index_type0
?
Variable/Adam_1
VariableV2*
dtype0*
_output_shapes
:	?*
shared_name *
_class
loc:@Variable*
	container *
shape:	?
?
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
:	?
x
Variable/Adam_1/readIdentityVariable/Adam_1*
_output_shapes
:	?*
T0*
_class
loc:@Variable
?
!Variable_1/Adam/Initializer/zerosConst*
_class
loc:@Variable_1*
valueB*    *
dtype0*
_output_shapes
:
?
Variable_1/Adam
VariableV2*
shared_name *
_class
loc:@Variable_1*
	container *
shape:*
dtype0*
_output_shapes
:
?
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes
:
u
Variable_1/Adam/readIdentityVariable_1/Adam*
T0*
_class
loc:@Variable_1*
_output_shapes
:
?
#Variable_1/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_1*
valueB*    *
dtype0*
_output_shapes
:
?
Variable_1/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_1*
	container *
shape:*
dtype0*
_output_shapes
:
?
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@Variable_1
y
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
_output_shapes
:*
T0*
_class
loc:@Variable_1
?
!Variable_2/Adam/Initializer/zerosConst*
_class
loc:@Variable_2*
valueB*    *
dtype0*
_output_shapes

:
?
Variable_2/Adam
VariableV2*
dtype0*
_output_shapes

:*
shared_name *
_class
loc:@Variable_2*
	container *
shape
:
?
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(*
_output_shapes

:
y
Variable_2/Adam/readIdentityVariable_2/Adam*
T0*
_class
loc:@Variable_2*
_output_shapes

:
?
#Variable_2/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_2*
valueB*    *
dtype0*
_output_shapes

:
?
Variable_2/Adam_1
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *
_class
loc:@Variable_2
?
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*
_class
loc:@Variable_2
}
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
T0*
_class
loc:@Variable_2*
_output_shapes

:
?
 b_fc_loc2/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*
_class
loc:@b_fc_loc2*
valueB*    
?
b_fc_loc2/Adam
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@b_fc_loc2
?
b_fc_loc2/Adam/AssignAssignb_fc_loc2/Adam b_fc_loc2/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@b_fc_loc2*
validate_shape(*
_output_shapes
:
r
b_fc_loc2/Adam/readIdentityb_fc_loc2/Adam*
T0*
_class
loc:@b_fc_loc2*
_output_shapes
:
?
"b_fc_loc2/Adam_1/Initializer/zerosConst*
_class
loc:@b_fc_loc2*
valueB*    *
dtype0*
_output_shapes
:
?
b_fc_loc2/Adam_1
VariableV2*
shared_name *
_class
loc:@b_fc_loc2*
	container *
shape:*
dtype0*
_output_shapes
:
?
b_fc_loc2/Adam_1/AssignAssignb_fc_loc2/Adam_1"b_fc_loc2/Adam_1/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@b_fc_loc2*
validate_shape(
v
b_fc_loc2/Adam_1/readIdentityb_fc_loc2/Adam_1*
T0*
_class
loc:@b_fc_loc2*
_output_shapes
:
?
!Variable_3/Adam/Initializer/zerosConst*
_class
loc:@Variable_3*%
valueB *    *
dtype0*&
_output_shapes
: 
?
Variable_3/Adam
VariableV2*
shared_name *
_class
loc:@Variable_3*
	container *
shape: *
dtype0*&
_output_shapes
: 
?
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*&
_output_shapes
: 
?
Variable_3/Adam/readIdentityVariable_3/Adam*
T0*
_class
loc:@Variable_3*&
_output_shapes
: 
?
#Variable_3/Adam_1/Initializer/zerosConst*
dtype0*&
_output_shapes
: *
_class
loc:@Variable_3*%
valueB *    
?
Variable_3/Adam_1
VariableV2*
_class
loc:@Variable_3*
	container *
shape: *
dtype0*&
_output_shapes
: *
shared_name 
?
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*&
_output_shapes
: 
?
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
T0*
_class
loc:@Variable_3*&
_output_shapes
: 
?
!Variable_4/Adam/Initializer/zerosConst*
_class
loc:@Variable_4*
valueB *    *
dtype0*
_output_shapes
: 
?
Variable_4/Adam
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@Variable_4*
	container 
?
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes
: *
use_locking(
u
Variable_4/Adam/readIdentityVariable_4/Adam*
_output_shapes
: *
T0*
_class
loc:@Variable_4
?
#Variable_4/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_4*
valueB *    *
dtype0*
_output_shapes
: 
?
Variable_4/Adam_1
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@Variable_4*
	container 
?
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Variable_4
y
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
T0*
_class
loc:@Variable_4*
_output_shapes
: 
?
1Variable_5/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Variable_5*%
valueB"          @   *
dtype0*
_output_shapes
:
?
'Variable_5/Adam/Initializer/zeros/ConstConst*
_class
loc:@Variable_5*
valueB
 *    *
dtype0*
_output_shapes
: 
?
!Variable_5/Adam/Initializer/zerosFill1Variable_5/Adam/Initializer/zeros/shape_as_tensor'Variable_5/Adam/Initializer/zeros/Const*
T0*
_class
loc:@Variable_5*

index_type0*&
_output_shapes
: @
?
Variable_5/Adam
VariableV2*
dtype0*&
_output_shapes
: @*
shared_name *
_class
loc:@Variable_5*
	container *
shape: @
?
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_5*
validate_shape(*&
_output_shapes
: @*
use_locking(
?
Variable_5/Adam/readIdentityVariable_5/Adam*&
_output_shapes
: @*
T0*
_class
loc:@Variable_5
?
3Variable_5/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
_class
loc:@Variable_5*%
valueB"          @   
?
)Variable_5/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@Variable_5*
valueB
 *    *
dtype0*
_output_shapes
: 
?
#Variable_5/Adam_1/Initializer/zerosFill3Variable_5/Adam_1/Initializer/zeros/shape_as_tensor)Variable_5/Adam_1/Initializer/zeros/Const*
_class
loc:@Variable_5*

index_type0*&
_output_shapes
: @*
T0
?
Variable_5/Adam_1
VariableV2*
dtype0*&
_output_shapes
: @*
shared_name *
_class
loc:@Variable_5*
	container *
shape: @
?
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*&
_output_shapes
: @
?
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*&
_output_shapes
: @*
T0*
_class
loc:@Variable_5
?
!Variable_6/Adam/Initializer/zerosConst*
_class
loc:@Variable_6*
valueB@*    *
dtype0*
_output_shapes
:@
?
Variable_6/Adam
VariableV2*
shared_name *
_class
loc:@Variable_6*
	container *
shape:@*
dtype0*
_output_shapes
:@
?
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(*
_output_shapes
:@
u
Variable_6/Adam/readIdentityVariable_6/Adam*
T0*
_class
loc:@Variable_6*
_output_shapes
:@
?
#Variable_6/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_6*
valueB@*    *
dtype0*
_output_shapes
:@
?
Variable_6/Adam_1
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *
_class
loc:@Variable_6*
	container *
shape:@
?
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(
y
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
T0*
_class
loc:@Variable_6*
_output_shapes
:@
?
1Variable_7/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Variable_7*
valueB"@     *
dtype0*
_output_shapes
:
?
'Variable_7/Adam/Initializer/zeros/ConstConst*
_class
loc:@Variable_7*
valueB
 *    *
dtype0*
_output_shapes
: 
?
!Variable_7/Adam/Initializer/zerosFill1Variable_7/Adam/Initializer/zeros/shape_as_tensor'Variable_7/Adam/Initializer/zeros/Const*
T0*
_class
loc:@Variable_7*

index_type0*
_output_shapes
:	?
?
Variable_7/Adam
VariableV2*
shared_name *
_class
loc:@Variable_7*
	container *
shape:	?*
dtype0*
_output_shapes
:	?
?
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes
:	?
z
Variable_7/Adam/readIdentityVariable_7/Adam*
T0*
_class
loc:@Variable_7*
_output_shapes
:	?
?
3Variable_7/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@Variable_7*
valueB"@     *
dtype0*
_output_shapes
:
?
)Variable_7/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@Variable_7*
valueB
 *    *
dtype0*
_output_shapes
: 
?
#Variable_7/Adam_1/Initializer/zerosFill3Variable_7/Adam_1/Initializer/zeros/shape_as_tensor)Variable_7/Adam_1/Initializer/zeros/Const*
_output_shapes
:	?*
T0*
_class
loc:@Variable_7*

index_type0
?
Variable_7/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_7*
	container *
shape:	?*
dtype0*
_output_shapes
:	?
?
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes
:	?
~
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
_output_shapes
:	?*
T0*
_class
loc:@Variable_7
?
!Variable_8/Adam/Initializer/zerosConst*
_class
loc:@Variable_8*
valueB*    *
dtype0*
_output_shapes
:
?
Variable_8/Adam
VariableV2*
shared_name *
_class
loc:@Variable_8*
	container *
shape:*
dtype0*
_output_shapes
:
?
Variable_8/Adam/AssignAssignVariable_8/Adam!Variable_8/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_8*
validate_shape(*
_output_shapes
:
u
Variable_8/Adam/readIdentityVariable_8/Adam*
T0*
_class
loc:@Variable_8*
_output_shapes
:
?
#Variable_8/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_8*
valueB*    *
dtype0*
_output_shapes
:
?
Variable_8/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_8*
	container *
shape:*
dtype0*
_output_shapes
:
?
Variable_8/Adam_1/AssignAssignVariable_8/Adam_1#Variable_8/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_8*
validate_shape(*
_output_shapes
:
y
Variable_8/Adam_1/readIdentityVariable_8/Adam_1*
T0*
_class
loc:@Variable_8*
_output_shapes
:
?
!Variable_9/Adam/Initializer/zerosConst*
_class
loc:@Variable_9*
valueB
*    *
dtype0*
_output_shapes

:

?
Variable_9/Adam
VariableV2*
shared_name *
_class
loc:@Variable_9*
	container *
shape
:
*
dtype0*
_output_shapes

:

?
Variable_9/Adam/AssignAssignVariable_9/Adam!Variable_9/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_9*
validate_shape(*
_output_shapes

:
*
use_locking(
y
Variable_9/Adam/readIdentityVariable_9/Adam*
T0*
_class
loc:@Variable_9*
_output_shapes

:

?
#Variable_9/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_9*
valueB
*    *
dtype0*
_output_shapes

:

?
Variable_9/Adam_1
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *
_class
loc:@Variable_9*
	container *
shape
:

?
Variable_9/Adam_1/AssignAssignVariable_9/Adam_1#Variable_9/Adam_1/Initializer/zeros*
_output_shapes

:
*
use_locking(*
T0*
_class
loc:@Variable_9*
validate_shape(
}
Variable_9/Adam_1/readIdentityVariable_9/Adam_1*
_class
loc:@Variable_9*
_output_shapes

:
*
T0
?
"Variable_10/Adam/Initializer/zerosConst*
_class
loc:@Variable_10*
valueB
*    *
dtype0*
_output_shapes
:

?
Variable_10/Adam
VariableV2*
shared_name *
_class
loc:@Variable_10*
	container *
shape:
*
dtype0*
_output_shapes
:

?
Variable_10/Adam/AssignAssignVariable_10/Adam"Variable_10/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_10*
validate_shape(*
_output_shapes
:

x
Variable_10/Adam/readIdentityVariable_10/Adam*
_class
loc:@Variable_10*
_output_shapes
:
*
T0
?
$Variable_10/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_10*
valueB
*    *
dtype0*
_output_shapes
:

?
Variable_10/Adam_1
VariableV2*
_class
loc:@Variable_10*
	container *
shape:
*
dtype0*
_output_shapes
:
*
shared_name 
?
Variable_10/Adam_1/AssignAssignVariable_10/Adam_1$Variable_10/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_10*
validate_shape(*
_output_shapes
:
*
use_locking(
|
Variable_10/Adam_1/readIdentityVariable_10/Adam_1*
T0*
_class
loc:@Variable_10*
_output_shapes
:

W
Adam/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *??8
O

Adam/beta1Const*
_output_shapes
: *
valueB
 *fff?*
dtype0
O

Adam/beta2Const*
valueB
 *w??*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *w?+2
?
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable*
use_nesterov( *
_output_shapes
:	?
?
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_1*
use_nesterov( *
_output_shapes
:*
use_locking( 
?
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_2*
use_nesterov( *
_output_shapes

:
?
Adam/update_b_fc_loc2/ApplyAdam	ApplyAdam	b_fc_loc2b_fc_loc2/Adamb_fc_loc2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_1_grad/tuple/control_dependency_1*
_output_shapes
:*
use_locking( *
T0*
_class
loc:@b_fc_loc2*
use_nesterov( 
?
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/Conv2D_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_3*
use_nesterov( *&
_output_shapes
: *
use_locking( 
?
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_2_grad/tuple/control_dependency_1*
_class
loc:@Variable_4*
use_nesterov( *
_output_shapes
: *
use_locking( *
T0
?
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_1_grad/tuple/control_dependency_1*
use_nesterov( *&
_output_shapes
: @*
use_locking( *
T0*
_class
loc:@Variable_5
?
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_3_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_6*
use_nesterov( *
_output_shapes
:@*
use_locking( 
?
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_2_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_7*
use_nesterov( *
_output_shapes
:	?*
use_locking( 
?
 Adam/update_Variable_8/ApplyAdam	ApplyAdam
Variable_8Variable_8/AdamVariable_8/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_4_grad/tuple/control_dependency_1*
_class
loc:@Variable_8*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0
?
 Adam/update_Variable_9/ApplyAdam	ApplyAdam
Variable_9Variable_9/AdamVariable_9/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_3_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_9*
use_nesterov( *
_output_shapes

:

?
!Adam/update_Variable_10/ApplyAdam	ApplyAdamVariable_10Variable_10/AdamVariable_10/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/y_conv_grad/tuple/control_dependency_1*
_output_shapes
:
*
use_locking( *
T0*
_class
loc:@Variable_10*
use_nesterov( 
?
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam"^Adam/update_Variable_10/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam ^Adam/update_b_fc_loc2/ApplyAdam*
T0*
_class
loc:@Variable*
_output_shapes
: 
?
Adam/AssignAssignbeta1_powerAdam/mul*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
?

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam"^Adam/update_Variable_10/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam ^Adam/update_b_fc_loc2/ApplyAdam*
T0*
_class
loc:@Variable*
_output_shapes
: 
?
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: *
use_locking( 
?
AdamNoOp^Adam/Assign^Adam/Assign_1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam"^Adam/update_Variable_10/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam ^Adam/update_b_fc_loc2/ApplyAdam
R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
w
ArgMaxArgMaxy_convArgMax/dimension*
output_type0	*#
_output_shapes
:?????????*

Tidx0*
T0
T
ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
w
ArgMax_1ArgMaxy_ArgMax_1/dimension*
output_type0	*#
_output_shapes
:?????????*

Tidx0*
T0
N
EqualEqualArgMaxArgMax_1*#
_output_shapes
:?????????*
T0	
`
CastCastEqual*
Truncate( *#
_output_shapes
:?????????*

DstT0*

SrcT0

Q
Const_6Const*
dtype0*
_output_shapes
:*
valueB: 
[
Mean_1MeanCastConst_6*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
?
initNoOp^Variable/Adam/Assign^Variable/Adam_1/Assign^Variable/Assign^Variable_1/Adam/Assign^Variable_1/Adam_1/Assign^Variable_1/Assign^Variable_10/Adam/Assign^Variable_10/Adam_1/Assign^Variable_10/Assign^Variable_2/Adam/Assign^Variable_2/Adam_1/Assign^Variable_2/Assign^Variable_3/Adam/Assign^Variable_3/Adam_1/Assign^Variable_3/Assign^Variable_4/Adam/Assign^Variable_4/Adam_1/Assign^Variable_4/Assign^Variable_5/Adam/Assign^Variable_5/Adam_1/Assign^Variable_5/Assign^Variable_6/Adam/Assign^Variable_6/Adam_1/Assign^Variable_6/Assign^Variable_7/Adam/Assign^Variable_7/Adam_1/Assign^Variable_7/Assign^Variable_8/Adam/Assign^Variable_8/Adam_1/Assign^Variable_8/Assign^Variable_9/Adam/Assign^Variable_9/Adam_1/Assign^Variable_9/Assign^b_fc_loc2/Adam/Assign^b_fc_loc2/Adam_1/Assign^b_fc_loc2/Assign^beta1_power/Assign^beta2_power/Assign
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0*
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
dtype0*
_output_shapes
: 
?
save/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_a719c9aeac73416ea3be75bb1f70078a/part*
dtype0
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
?
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
?
save/SaveV2/tensor_namesConst"/device:CPU:0*?
value?B?&BVariableBVariable/AdamBVariable/Adam_1B
Variable_1BVariable_1/AdamBVariable_1/Adam_1BVariable_10BVariable_10/AdamBVariable_10/Adam_1B
Variable_2BVariable_2/AdamBVariable_2/Adam_1B
Variable_3BVariable_3/AdamBVariable_3/Adam_1B
Variable_4BVariable_4/AdamBVariable_4/Adam_1B
Variable_5BVariable_5/AdamBVariable_5/Adam_1B
Variable_6BVariable_6/AdamBVariable_6/Adam_1B
Variable_7BVariable_7/AdamBVariable_7/Adam_1B
Variable_8BVariable_8/AdamBVariable_8/Adam_1B
Variable_9BVariable_9/AdamBVariable_9/Adam_1B	b_fc_loc2Bb_fc_loc2/AdamBb_fc_loc2/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:&
?
save/SaveV2/shape_and_slicesConst"/device:CPU:0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&
?
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesVariableVariable/AdamVariable/Adam_1
Variable_1Variable_1/AdamVariable_1/Adam_1Variable_10Variable_10/AdamVariable_10/Adam_1
Variable_2Variable_2/AdamVariable_2/Adam_1
Variable_3Variable_3/AdamVariable_3/Adam_1
Variable_4Variable_4/AdamVariable_4/Adam_1
Variable_5Variable_5/AdamVariable_5/Adam_1
Variable_6Variable_6/AdamVariable_6/Adam_1
Variable_7Variable_7/AdamVariable_7/Adam_1
Variable_8Variable_8/AdamVariable_8/Adam_1
Variable_9Variable_9/AdamVariable_9/Adam_1	b_fc_loc2b_fc_loc2/Adamb_fc_loc2/Adam_1beta1_powerbeta2_power"/device:CPU:0*4
dtypes*
(2&
?
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
_output_shapes
: *
T0*'
_class
loc:@save/ShardedFilename
?
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:
?
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(
?
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
?
save/RestoreV2/tensor_namesConst"/device:CPU:0*?
value?B?&BVariableBVariable/AdamBVariable/Adam_1B
Variable_1BVariable_1/AdamBVariable_1/Adam_1BVariable_10BVariable_10/AdamBVariable_10/Adam_1B
Variable_2BVariable_2/AdamBVariable_2/Adam_1B
Variable_3BVariable_3/AdamBVariable_3/Adam_1B
Variable_4BVariable_4/AdamBVariable_4/Adam_1B
Variable_5BVariable_5/AdamBVariable_5/Adam_1B
Variable_6BVariable_6/AdamBVariable_6/Adam_1B
Variable_7BVariable_7/AdamBVariable_7/Adam_1B
Variable_8BVariable_8/AdamBVariable_8/Adam_1B
Variable_9BVariable_9/AdamBVariable_9/Adam_1B	b_fc_loc2Bb_fc_loc2/AdamBb_fc_loc2/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:&
?
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::*4
dtypes*
(2&
?
save/AssignAssignVariablesave/RestoreV2*
validate_shape(*
_output_shapes
:	?*
use_locking(*
T0*
_class
loc:@Variable
?
save/Assign_1AssignVariable/Adamsave/RestoreV2:1*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
:	?
?
save/Assign_2AssignVariable/Adam_1save/RestoreV2:2*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
:	?
?
save/Assign_3Assign
Variable_1save/RestoreV2:3*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes
:
?
save/Assign_4AssignVariable_1/Adamsave/RestoreV2:4*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
?
save/Assign_5AssignVariable_1/Adam_1save/RestoreV2:5*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
?
save/Assign_6AssignVariable_10save/RestoreV2:6*
use_locking(*
T0*
_class
loc:@Variable_10*
validate_shape(*
_output_shapes
:

?
save/Assign_7AssignVariable_10/Adamsave/RestoreV2:7*
use_locking(*
T0*
_class
loc:@Variable_10*
validate_shape(*
_output_shapes
:

?
save/Assign_8AssignVariable_10/Adam_1save/RestoreV2:8*
use_locking(*
T0*
_class
loc:@Variable_10*
validate_shape(*
_output_shapes
:

?
save/Assign_9Assign
Variable_2save/RestoreV2:9*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*
_class
loc:@Variable_2
?
save/Assign_10AssignVariable_2/Adamsave/RestoreV2:10*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(*
_output_shapes

:
?
save/Assign_11AssignVariable_2/Adam_1save/RestoreV2:11*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(*
_output_shapes

:
?
save/Assign_12Assign
Variable_3save/RestoreV2:12*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Variable_3
?
save/Assign_13AssignVariable_3/Adamsave/RestoreV2:13*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*&
_output_shapes
: 
?
save/Assign_14AssignVariable_3/Adam_1save/RestoreV2:14*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*&
_output_shapes
: 
?
save/Assign_15Assign
Variable_4save/RestoreV2:15*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes
: 
?
save/Assign_16AssignVariable_4/Adamsave/RestoreV2:16*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes
: 
?
save/Assign_17AssignVariable_4/Adam_1save/RestoreV2:17*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes
: 
?
save/Assign_18Assign
Variable_5save/RestoreV2:18*
T0*
_class
loc:@Variable_5*
validate_shape(*&
_output_shapes
: @*
use_locking(
?
save/Assign_19AssignVariable_5/Adamsave/RestoreV2:19*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*&
_output_shapes
: @
?
save/Assign_20AssignVariable_5/Adam_1save/RestoreV2:20*
T0*
_class
loc:@Variable_5*
validate_shape(*&
_output_shapes
: @*
use_locking(
?
save/Assign_21Assign
Variable_6save/RestoreV2:21*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(*
_output_shapes
:@
?
save/Assign_22AssignVariable_6/Adamsave/RestoreV2:22*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(*
_output_shapes
:@
?
save/Assign_23AssignVariable_6/Adam_1save/RestoreV2:23*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(*
_output_shapes
:@
?
save/Assign_24Assign
Variable_7save/RestoreV2:24*
validate_shape(*
_output_shapes
:	?*
use_locking(*
T0*
_class
loc:@Variable_7
?
save/Assign_25AssignVariable_7/Adamsave/RestoreV2:25*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes
:	?*
use_locking(
?
save/Assign_26AssignVariable_7/Adam_1save/RestoreV2:26*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes
:	?
?
save/Assign_27Assign
Variable_8save/RestoreV2:27*
use_locking(*
T0*
_class
loc:@Variable_8*
validate_shape(*
_output_shapes
:
?
save/Assign_28AssignVariable_8/Adamsave/RestoreV2:28*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@Variable_8
?
save/Assign_29AssignVariable_8/Adam_1save/RestoreV2:29*
T0*
_class
loc:@Variable_8*
validate_shape(*
_output_shapes
:*
use_locking(
?
save/Assign_30Assign
Variable_9save/RestoreV2:30*
_class
loc:@Variable_9*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0
?
save/Assign_31AssignVariable_9/Adamsave/RestoreV2:31*
_output_shapes

:
*
use_locking(*
T0*
_class
loc:@Variable_9*
validate_shape(
?
save/Assign_32AssignVariable_9/Adam_1save/RestoreV2:32*
use_locking(*
T0*
_class
loc:@Variable_9*
validate_shape(*
_output_shapes

:

?
save/Assign_33Assign	b_fc_loc2save/RestoreV2:33*
T0*
_class
loc:@b_fc_loc2*
validate_shape(*
_output_shapes
:*
use_locking(
?
save/Assign_34Assignb_fc_loc2/Adamsave/RestoreV2:34*
use_locking(*
T0*
_class
loc:@b_fc_loc2*
validate_shape(*
_output_shapes
:
?
save/Assign_35Assignb_fc_loc2/Adam_1save/RestoreV2:35*
use_locking(*
T0*
_class
loc:@b_fc_loc2*
validate_shape(*
_output_shapes
:
?
save/Assign_36Assignbeta1_powersave/RestoreV2:36*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: *
use_locking(
?
save/Assign_37Assignbeta2_powersave/RestoreV2:37*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
?
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard"&<
save/Const:0save/Identity:0save/restore_all (5 @F8"?
trainable_variables??
D

Variable:0Variable/AssignVariable/read:02truncated_normal:08
?
Variable_1:0Variable_1/AssignVariable_1/read:02Const:08
L
Variable_2:0Variable_2/AssignVariable_2/read:02truncated_normal_1:08
N
b_fc_loc2:0b_fc_loc2/Assignb_fc_loc2/read:02b_fc_loc2/initial_value:08
L
Variable_3:0Variable_3/AssignVariable_3/read:02truncated_normal_2:08
A
Variable_4:0Variable_4/AssignVariable_4/read:02	Const_1:08
L
Variable_5:0Variable_5/AssignVariable_5/read:02truncated_normal_3:08
A
Variable_6:0Variable_6/AssignVariable_6/read:02	Const_2:08
L
Variable_7:0Variable_7/AssignVariable_7/read:02truncated_normal_4:08
A
Variable_8:0Variable_8/AssignVariable_8/read:02	Const_3:08
L
Variable_9:0Variable_9/AssignVariable_9/read:02truncated_normal_5:08
D
Variable_10:0Variable_10/AssignVariable_10/read:02	Const_4:08"
train_op

Adam"?
	variables??
D

Variable:0Variable/AssignVariable/read:02truncated_normal:08
?
Variable_1:0Variable_1/AssignVariable_1/read:02Const:08
L
Variable_2:0Variable_2/AssignVariable_2/read:02truncated_normal_1:08
N
b_fc_loc2:0b_fc_loc2/Assignb_fc_loc2/read:02b_fc_loc2/initial_value:08
L
Variable_3:0Variable_3/AssignVariable_3/read:02truncated_normal_2:08
A
Variable_4:0Variable_4/AssignVariable_4/read:02	Const_1:08
L
Variable_5:0Variable_5/AssignVariable_5/read:02truncated_normal_3:08
A
Variable_6:0Variable_6/AssignVariable_6/read:02	Const_2:08
L
Variable_7:0Variable_7/AssignVariable_7/read:02truncated_normal_4:08
A
Variable_8:0Variable_8/AssignVariable_8/read:02	Const_3:08
L
Variable_9:0Variable_9/AssignVariable_9/read:02truncated_normal_5:08
D
Variable_10:0Variable_10/AssignVariable_10/read:02	Const_4:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
`
Variable/Adam:0Variable/Adam/AssignVariable/Adam/read:02!Variable/Adam/Initializer/zeros:0
h
Variable/Adam_1:0Variable/Adam_1/AssignVariable/Adam_1/read:02#Variable/Adam_1/Initializer/zeros:0
h
Variable_1/Adam:0Variable_1/Adam/AssignVariable_1/Adam/read:02#Variable_1/Adam/Initializer/zeros:0
p
Variable_1/Adam_1:0Variable_1/Adam_1/AssignVariable_1/Adam_1/read:02%Variable_1/Adam_1/Initializer/zeros:0
h
Variable_2/Adam:0Variable_2/Adam/AssignVariable_2/Adam/read:02#Variable_2/Adam/Initializer/zeros:0
p
Variable_2/Adam_1:0Variable_2/Adam_1/AssignVariable_2/Adam_1/read:02%Variable_2/Adam_1/Initializer/zeros:0
d
b_fc_loc2/Adam:0b_fc_loc2/Adam/Assignb_fc_loc2/Adam/read:02"b_fc_loc2/Adam/Initializer/zeros:0
l
b_fc_loc2/Adam_1:0b_fc_loc2/Adam_1/Assignb_fc_loc2/Adam_1/read:02$b_fc_loc2/Adam_1/Initializer/zeros:0
h
Variable_3/Adam:0Variable_3/Adam/AssignVariable_3/Adam/read:02#Variable_3/Adam/Initializer/zeros:0
p
Variable_3/Adam_1:0Variable_3/Adam_1/AssignVariable_3/Adam_1/read:02%Variable_3/Adam_1/Initializer/zeros:0
h
Variable_4/Adam:0Variable_4/Adam/AssignVariable_4/Adam/read:02#Variable_4/Adam/Initializer/zeros:0
p
Variable_4/Adam_1:0Variable_4/Adam_1/AssignVariable_4/Adam_1/read:02%Variable_4/Adam_1/Initializer/zeros:0
h
Variable_5/Adam:0Variable_5/Adam/AssignVariable_5/Adam/read:02#Variable_5/Adam/Initializer/zeros:0
p
Variable_5/Adam_1:0Variable_5/Adam_1/AssignVariable_5/Adam_1/read:02%Variable_5/Adam_1/Initializer/zeros:0
h
Variable_6/Adam:0Variable_6/Adam/AssignVariable_6/Adam/read:02#Variable_6/Adam/Initializer/zeros:0
p
Variable_6/Adam_1:0Variable_6/Adam_1/AssignVariable_6/Adam_1/read:02%Variable_6/Adam_1/Initializer/zeros:0
h
Variable_7/Adam:0Variable_7/Adam/AssignVariable_7/Adam/read:02#Variable_7/Adam/Initializer/zeros:0
p
Variable_7/Adam_1:0Variable_7/Adam_1/AssignVariable_7/Adam_1/read:02%Variable_7/Adam_1/Initializer/zeros:0
h
Variable_8/Adam:0Variable_8/Adam/AssignVariable_8/Adam/read:02#Variable_8/Adam/Initializer/zeros:0
p
Variable_8/Adam_1:0Variable_8/Adam_1/AssignVariable_8/Adam_1/read:02%Variable_8/Adam_1/Initializer/zeros:0
h
Variable_9/Adam:0Variable_9/Adam/AssignVariable_9/Adam/read:02#Variable_9/Adam/Initializer/zeros:0
p
Variable_9/Adam_1:0Variable_9/Adam_1/AssignVariable_9/Adam_1/read:02%Variable_9/Adam_1/Initializer/zeros:0
l
Variable_10/Adam:0Variable_10/Adam/AssignVariable_10/Adam/read:02$Variable_10/Adam/Initializer/zeros:0
t
Variable_10/Adam_1:0Variable_10/Adam_1/AssignVariable_10/Adam_1/read:02&Variable_10/Adam_1/Initializer/zeros:0*d
serving_defaultQ
$
input
x:0??????????)
output
y_conv:0?????????
