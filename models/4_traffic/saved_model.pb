��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.2.02unknown8��
�
conv2d_52/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_52/kernel
}
$conv2d_52/kernel/Read/ReadVariableOpReadVariableOpconv2d_52/kernel*&
_output_shapes
: *
dtype0
t
conv2d_52/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_52/bias
m
"conv2d_52/bias/Read/ReadVariableOpReadVariableOpconv2d_52/bias*
_output_shapes
: *
dtype0
�
conv2d_53/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_53/kernel
}
$conv2d_53/kernel/Read/ReadVariableOpReadVariableOpconv2d_53/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_53/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_53/bias
m
"conv2d_53/bias/Read/ReadVariableOpReadVariableOpconv2d_53/bias*
_output_shapes
: *
dtype0
�
batch_normalization_24/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_24/gamma
�
0batch_normalization_24/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_24/gamma*
_output_shapes
: *
dtype0
�
batch_normalization_24/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_24/beta
�
/batch_normalization_24/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_24/beta*
_output_shapes
: *
dtype0
�
"batch_normalization_24/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_24/moving_mean
�
6batch_normalization_24/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_24/moving_mean*
_output_shapes
: *
dtype0
�
&batch_normalization_24/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_24/moving_variance
�
:batch_normalization_24/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_24/moving_variance*
_output_shapes
: *
dtype0
�
conv2d_54/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_54/kernel
}
$conv2d_54/kernel/Read/ReadVariableOpReadVariableOpconv2d_54/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_54/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_54/bias
m
"conv2d_54/bias/Read/ReadVariableOpReadVariableOpconv2d_54/bias*
_output_shapes
:@*
dtype0
�
conv2d_55/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_55/kernel
}
$conv2d_55/kernel/Read/ReadVariableOpReadVariableOpconv2d_55/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_55/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_55/bias
m
"conv2d_55/bias/Read/ReadVariableOpReadVariableOpconv2d_55/bias*
_output_shapes
:@*
dtype0
�
batch_normalization_25/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_25/gamma
�
0batch_normalization_25/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_25/gamma*
_output_shapes
:@*
dtype0
�
batch_normalization_25/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_25/beta
�
/batch_normalization_25/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_25/beta*
_output_shapes
:@*
dtype0
�
"batch_normalization_25/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_25/moving_mean
�
6batch_normalization_25/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_25/moving_mean*
_output_shapes
:@*
dtype0
�
&batch_normalization_25/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_25/moving_variance
�
:batch_normalization_25/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_25/moving_variance*
_output_shapes
:@*
dtype0
|
dense_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_26/kernel
u
#dense_26/kernel/Read/ReadVariableOpReadVariableOpdense_26/kernel* 
_output_shapes
:
��*
dtype0
s
dense_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_26/bias
l
!dense_26/bias/Read/ReadVariableOpReadVariableOpdense_26/bias*
_output_shapes	
:�*
dtype0
{
dense_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�+* 
shared_namedense_27/kernel
t
#dense_27/kernel/Read/ReadVariableOpReadVariableOpdense_27/kernel*
_output_shapes
:	�+*
dtype0
r
dense_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:+*
shared_namedense_27/bias
k
!dense_27/bias/Read/ReadVariableOpReadVariableOpdense_27/bias*
_output_shapes
:+*
dtype0

NoOpNoOp
�;
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�;
value�;B�; B�;
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer-7
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer-12
layer_with_weights-7
layer-13
	optimizer

signatures
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
 	keras_api
R
!regularization_losses
"trainable_variables
#	variables
$	keras_api
R
%regularization_losses
&trainable_variables
'	variables
(	keras_api
�
)axis
	*gamma
+beta
,moving_mean
-moving_variance
.regularization_losses
/trainable_variables
0	variables
1	keras_api
h

2kernel
3bias
4regularization_losses
5trainable_variables
6	variables
7	keras_api
h

8kernel
9bias
:regularization_losses
;trainable_variables
<	variables
=	keras_api
R
>regularization_losses
?trainable_variables
@	variables
A	keras_api
R
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
�
Faxis
	Ggamma
Hbeta
Imoving_mean
Jmoving_variance
Kregularization_losses
Ltrainable_variables
M	variables
N	keras_api
R
Oregularization_losses
Ptrainable_variables
Q	variables
R	keras_api
h

Skernel
Tbias
Uregularization_losses
Vtrainable_variables
W	variables
X	keras_api
R
Yregularization_losses
Ztrainable_variables
[	variables
\	keras_api
h

]kernel
^bias
_regularization_losses
`trainable_variables
a	variables
b	keras_api
 
 
 
v
0
1
2
3
*4
+5
26
37
88
99
G10
H11
S12
T13
]14
^15
�
0
1
2
3
*4
+5
,6
-7
28
39
810
911
G12
H13
I14
J15
S16
T17
]18
^19
�
regularization_losses
clayer_metrics
dmetrics
trainable_variables
enon_trainable_variables
flayer_regularization_losses
	variables

glayers
\Z
VARIABLE_VALUEconv2d_52/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_52/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
regularization_losses
hlayer_metrics
imetrics
trainable_variables
jnon_trainable_variables
klayer_regularization_losses
	variables

llayers
\Z
VARIABLE_VALUEconv2d_53/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_53/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
regularization_losses
mlayer_metrics
nmetrics
trainable_variables
onon_trainable_variables
player_regularization_losses
	variables

qlayers
 
 
 
�
!regularization_losses
rlayer_metrics
smetrics
"trainable_variables
tnon_trainable_variables
ulayer_regularization_losses
#	variables

vlayers
 
 
 
�
%regularization_losses
wlayer_metrics
xmetrics
&trainable_variables
ynon_trainable_variables
zlayer_regularization_losses
'	variables

{layers
 
ge
VARIABLE_VALUEbatch_normalization_24/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_24/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_24/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_24/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

*0
+1

*0
+1
,2
-3
�
.regularization_losses
|layer_metrics
}metrics
/trainable_variables
~non_trainable_variables
layer_regularization_losses
0	variables
�layers
\Z
VARIABLE_VALUEconv2d_54/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_54/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

20
31

20
31
�
4regularization_losses
�layer_metrics
�metrics
5trainable_variables
�non_trainable_variables
 �layer_regularization_losses
6	variables
�layers
\Z
VARIABLE_VALUEconv2d_55/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_55/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

80
91

80
91
�
:regularization_losses
�layer_metrics
�metrics
;trainable_variables
�non_trainable_variables
 �layer_regularization_losses
<	variables
�layers
 
 
 
�
>regularization_losses
�layer_metrics
�metrics
?trainable_variables
�non_trainable_variables
 �layer_regularization_losses
@	variables
�layers
 
 
 
�
Bregularization_losses
�layer_metrics
�metrics
Ctrainable_variables
�non_trainable_variables
 �layer_regularization_losses
D	variables
�layers
 
ge
VARIABLE_VALUEbatch_normalization_25/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_25/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_25/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_25/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

G0
H1

G0
H1
I2
J3
�
Kregularization_losses
�layer_metrics
�metrics
Ltrainable_variables
�non_trainable_variables
 �layer_regularization_losses
M	variables
�layers
 
 
 
�
Oregularization_losses
�layer_metrics
�metrics
Ptrainable_variables
�non_trainable_variables
 �layer_regularization_losses
Q	variables
�layers
[Y
VARIABLE_VALUEdense_26/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_26/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

S0
T1

S0
T1
�
Uregularization_losses
�layer_metrics
�metrics
Vtrainable_variables
�non_trainable_variables
 �layer_regularization_losses
W	variables
�layers
 
 
 
�
Yregularization_losses
�layer_metrics
�metrics
Ztrainable_variables
�non_trainable_variables
 �layer_regularization_losses
[	variables
�layers
[Y
VARIABLE_VALUEdense_27/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_27/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

]0
^1

]0
^1
�
_regularization_losses
�layer_metrics
�metrics
`trainable_variables
�non_trainable_variables
 �layer_regularization_losses
a	variables
�layers
 
 

,0
-1
I2
J3
 
f
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

,0
-1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

I0
J1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
�
serving_default_input_36Placeholder*/
_output_shapes
:���������*
dtype0*$
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_36conv2d_52/kernelconv2d_52/biasconv2d_53/kernelconv2d_53/biasbatch_normalization_24/gammabatch_normalization_24/beta"batch_normalization_24/moving_mean&batch_normalization_24/moving_varianceconv2d_54/kernelconv2d_54/biasconv2d_55/kernelconv2d_55/biasbatch_normalization_25/gammabatch_normalization_25/beta"batch_normalization_25/moving_mean&batch_normalization_25/moving_variancedense_26/kerneldense_26/biasdense_27/kerneldense_27/bias* 
Tin
2*
Tout
2*'
_output_shapes
:���������+*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*/
f*R(
&__inference_signature_wrapper_18034466
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_52/kernel/Read/ReadVariableOp"conv2d_52/bias/Read/ReadVariableOp$conv2d_53/kernel/Read/ReadVariableOp"conv2d_53/bias/Read/ReadVariableOp0batch_normalization_24/gamma/Read/ReadVariableOp/batch_normalization_24/beta/Read/ReadVariableOp6batch_normalization_24/moving_mean/Read/ReadVariableOp:batch_normalization_24/moving_variance/Read/ReadVariableOp$conv2d_54/kernel/Read/ReadVariableOp"conv2d_54/bias/Read/ReadVariableOp$conv2d_55/kernel/Read/ReadVariableOp"conv2d_55/bias/Read/ReadVariableOp0batch_normalization_25/gamma/Read/ReadVariableOp/batch_normalization_25/beta/Read/ReadVariableOp6batch_normalization_25/moving_mean/Read/ReadVariableOp:batch_normalization_25/moving_variance/Read/ReadVariableOp#dense_26/kernel/Read/ReadVariableOp!dense_26/bias/Read/ReadVariableOp#dense_27/kernel/Read/ReadVariableOp!dense_27/bias/Read/ReadVariableOpConst*!
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8**
f%R#
!__inference__traced_save_18035363
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_52/kernelconv2d_52/biasconv2d_53/kernelconv2d_53/biasbatch_normalization_24/gammabatch_normalization_24/beta"batch_normalization_24/moving_mean&batch_normalization_24/moving_varianceconv2d_54/kernelconv2d_54/biasconv2d_55/kernelconv2d_55/biasbatch_normalization_25/gammabatch_normalization_25/beta"batch_normalization_25/moving_mean&batch_normalization_25/moving_variancedense_26/kerneldense_26/biasdense_27/kerneldense_27/bias* 
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*-
f(R&
$__inference__traced_restore_18035435��
�
g
H__inference_dropout_41_layer_call_and_return_conditional_losses_18035241

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
I
-__inference_dropout_40_layer_call_fn_18035031

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_dropout_40_layer_call_and_return_conditional_losses_180339292
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�$
�
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_18035074

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
0__inference_sequential_13_layer_call_fn_18034770

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*'
_output_shapes
:���������+*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_sequential_13_layer_call_and_return_conditional_losses_180342492
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������+2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:���������::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
I
-__inference_flatten_13_layer_call_fn_18035204

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_flatten_13_layer_call_and_return_conditional_losses_180340322
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
,__inference_conv2d_55_layer_call_fn_18033634

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_conv2d_55_layer_call_and_return_conditional_losses_180336242
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
F__inference_dense_27_layer_call_and_return_conditional_losses_18034113

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�+*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������+2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:+*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������+2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������+2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:���������+2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
9__inference_batch_normalization_24_layer_call_fn_18034991

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*]
fXRV
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_180338472
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:��������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�I
�
K__inference_sequential_13_layer_call_and_return_conditional_losses_18034130
input_36
conv2d_52_18033776
conv2d_52_18033778
conv2d_53_18033781
conv2d_53_18033783#
batch_normalization_24_18033892#
batch_normalization_24_18033894#
batch_normalization_24_18033896#
batch_normalization_24_18033898
conv2d_54_18033901
conv2d_54_18033903
conv2d_55_18033906
conv2d_55_18033908#
batch_normalization_25_18034017#
batch_normalization_25_18034019#
batch_normalization_25_18034021#
batch_normalization_25_18034023
dense_26_18034067
dense_26_18034069
dense_27_18034124
dense_27_18034126
identity��.batch_normalization_24/StatefulPartitionedCall�.batch_normalization_25/StatefulPartitionedCall�!conv2d_52/StatefulPartitionedCall�!conv2d_53/StatefulPartitionedCall�!conv2d_54/StatefulPartitionedCall�!conv2d_55/StatefulPartitionedCall� dense_26/StatefulPartitionedCall� dense_27/StatefulPartitionedCall�"dropout_39/StatefulPartitionedCall�"dropout_40/StatefulPartitionedCall�"dropout_41/StatefulPartitionedCall�
!conv2d_52/StatefulPartitionedCallStatefulPartitionedCallinput_36conv2d_52_18033776conv2d_52_18033778*
Tin
2*
Tout
2*/
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_conv2d_52_layer_call_and_return_conditional_losses_180334052#
!conv2d_52/StatefulPartitionedCall�
!conv2d_53/StatefulPartitionedCallStatefulPartitionedCall*conv2d_52/StatefulPartitionedCall:output:0conv2d_53_18033781conv2d_53_18033783*
Tin
2*
Tout
2*/
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_conv2d_53_layer_call_and_return_conditional_losses_180334322#
!conv2d_53/StatefulPartitionedCall�
 max_pooling2d_26/PartitionedCallPartitionedCall*conv2d_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_180334482"
 max_pooling2d_26/PartitionedCall�
"dropout_39/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_26/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_dropout_39_layer_call_and_return_conditional_losses_180337992$
"dropout_39/StatefulPartitionedCall�
.batch_normalization_24/StatefulPartitionedCallStatefulPartitionedCall+dropout_39/StatefulPartitionedCall:output:0batch_normalization_24_18033892batch_normalization_24_18033894batch_normalization_24_18033896batch_normalization_24_18033898*
Tin	
2*
Tout
2*/
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*]
fXRV
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_1803384720
.batch_normalization_24/StatefulPartitionedCall�
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_24/StatefulPartitionedCall:output:0conv2d_54_18033901conv2d_54_18033903*
Tin
2*
Tout
2*/
_output_shapes
:���������		@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_conv2d_54_layer_call_and_return_conditional_losses_180335972#
!conv2d_54/StatefulPartitionedCall�
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall*conv2d_54/StatefulPartitionedCall:output:0conv2d_55_18033906conv2d_55_18033908*
Tin
2*
Tout
2*/
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_conv2d_55_layer_call_and_return_conditional_losses_180336242#
!conv2d_55/StatefulPartitionedCall�
 max_pooling2d_27/PartitionedCallPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_max_pooling2d_27_layer_call_and_return_conditional_losses_180336402"
 max_pooling2d_27/PartitionedCall�
"dropout_40/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_27/PartitionedCall:output:0#^dropout_39/StatefulPartitionedCall*
Tin
2*
Tout
2*/
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_dropout_40_layer_call_and_return_conditional_losses_180339242$
"dropout_40/StatefulPartitionedCall�
.batch_normalization_25/StatefulPartitionedCallStatefulPartitionedCall+dropout_40/StatefulPartitionedCall:output:0batch_normalization_25_18034017batch_normalization_25_18034019batch_normalization_25_18034021batch_normalization_25_18034023*
Tin	
2*
Tout
2*/
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*]
fXRV
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_1803397220
.batch_normalization_25/StatefulPartitionedCall�
flatten_13/PartitionedCallPartitionedCall7batch_normalization_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_flatten_13_layer_call_and_return_conditional_losses_180340322
flatten_13/PartitionedCall�
 dense_26/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_26_18034067dense_26_18034069*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dense_26_layer_call_and_return_conditional_losses_180340562"
 dense_26/StatefulPartitionedCall�
"dropout_41/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0#^dropout_40/StatefulPartitionedCall*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_dropout_41_layer_call_and_return_conditional_losses_180340842$
"dropout_41/StatefulPartitionedCall�
 dense_27/StatefulPartitionedCallStatefulPartitionedCall+dropout_41/StatefulPartitionedCall:output:0dense_27_18034124dense_27_18034126*
Tin
2*
Tout
2*'
_output_shapes
:���������+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dense_27_layer_call_and_return_conditional_losses_180341132"
 dense_27/StatefulPartitionedCall�
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0/^batch_normalization_24/StatefulPartitionedCall/^batch_normalization_25/StatefulPartitionedCall"^conv2d_52/StatefulPartitionedCall"^conv2d_53/StatefulPartitionedCall"^conv2d_54/StatefulPartitionedCall"^conv2d_55/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall#^dropout_39/StatefulPartitionedCall#^dropout_40/StatefulPartitionedCall#^dropout_41/StatefulPartitionedCall*
T0*'
_output_shapes
:���������+2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:���������::::::::::::::::::::2`
.batch_normalization_24/StatefulPartitionedCall.batch_normalization_24/StatefulPartitionedCall2`
.batch_normalization_25/StatefulPartitionedCall.batch_normalization_25/StatefulPartitionedCall2F
!conv2d_52/StatefulPartitionedCall!conv2d_52/StatefulPartitionedCall2F
!conv2d_53/StatefulPartitionedCall!conv2d_53/StatefulPartitionedCall2F
!conv2d_54/StatefulPartitionedCall!conv2d_54/StatefulPartitionedCall2F
!conv2d_55/StatefulPartitionedCall!conv2d_55/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2H
"dropout_39/StatefulPartitionedCall"dropout_39/StatefulPartitionedCall2H
"dropout_40/StatefulPartitionedCall"dropout_40/StatefulPartitionedCall2H
"dropout_41/StatefulPartitionedCall"dropout_41/StatefulPartitionedCall:Y U
/
_output_shapes
:���������
"
_user_specified_name
input_36:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�k
�	
K__inference_sequential_13_layer_call_and_return_conditional_losses_18034725

inputs,
(conv2d_52_conv2d_readvariableop_resource-
)conv2d_52_biasadd_readvariableop_resource,
(conv2d_53_conv2d_readvariableop_resource-
)conv2d_53_biasadd_readvariableop_resource2
.batch_normalization_24_readvariableop_resource4
0batch_normalization_24_readvariableop_1_resourceC
?batch_normalization_24_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_54_conv2d_readvariableop_resource-
)conv2d_54_biasadd_readvariableop_resource,
(conv2d_55_conv2d_readvariableop_resource-
)conv2d_55_biasadd_readvariableop_resource2
.batch_normalization_25_readvariableop_resource4
0batch_normalization_25_readvariableop_1_resourceC
?batch_normalization_25_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource+
'dense_26_matmul_readvariableop_resource,
(dense_26_biasadd_readvariableop_resource+
'dense_27_matmul_readvariableop_resource,
(dense_27_biasadd_readvariableop_resource
identity��
conv2d_52/Conv2D/ReadVariableOpReadVariableOp(conv2d_52_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_52/Conv2D/ReadVariableOp�
conv2d_52/Conv2DConv2Dinputs'conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
conv2d_52/Conv2D�
 conv2d_52/BiasAdd/ReadVariableOpReadVariableOp)conv2d_52_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_52/BiasAdd/ReadVariableOp�
conv2d_52/BiasAddBiasAddconv2d_52/Conv2D:output:0(conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
conv2d_52/BiasAdd�
conv2d_52/SigmoidSigmoidconv2d_52/BiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
conv2d_52/Sigmoid�
conv2d_52/mulMulconv2d_52/BiasAdd:output:0conv2d_52/Sigmoid:y:0*
T0*/
_output_shapes
:��������� 2
conv2d_52/mul�
conv2d_52/IdentityIdentityconv2d_52/mul:z:0*
T0*/
_output_shapes
:��������� 2
conv2d_52/Identity�
conv2d_52/IdentityN	IdentityNconv2d_52/mul:z:0conv2d_52/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-18034628*J
_output_shapes8
6:��������� :��������� 2
conv2d_52/IdentityN�
conv2d_53/Conv2D/ReadVariableOpReadVariableOp(conv2d_53_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_53/Conv2D/ReadVariableOp�
conv2d_53/Conv2DConv2Dconv2d_52/IdentityN:output:0'conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
conv2d_53/Conv2D�
 conv2d_53/BiasAdd/ReadVariableOpReadVariableOp)conv2d_53_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_53/BiasAdd/ReadVariableOp�
conv2d_53/BiasAddBiasAddconv2d_53/Conv2D:output:0(conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
conv2d_53/BiasAdd�
conv2d_53/SigmoidSigmoidconv2d_53/BiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
conv2d_53/Sigmoid�
conv2d_53/mulMulconv2d_53/BiasAdd:output:0conv2d_53/Sigmoid:y:0*
T0*/
_output_shapes
:��������� 2
conv2d_53/mul�
conv2d_53/IdentityIdentityconv2d_53/mul:z:0*
T0*/
_output_shapes
:��������� 2
conv2d_53/Identity�
conv2d_53/IdentityN	IdentityNconv2d_53/mul:z:0conv2d_53/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-18034640*J
_output_shapes8
6:��������� :��������� 2
conv2d_53/IdentityN�
max_pooling2d_26/MaxPoolMaxPoolconv2d_53/IdentityN:output:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2
max_pooling2d_26/MaxPool�
dropout_39/IdentityIdentity!max_pooling2d_26/MaxPool:output:0*
T0*/
_output_shapes
:��������� 2
dropout_39/Identity�
%batch_normalization_24/ReadVariableOpReadVariableOp.batch_normalization_24_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_24/ReadVariableOp�
'batch_normalization_24/ReadVariableOp_1ReadVariableOp0batch_normalization_24_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_24/ReadVariableOp_1�
6batch_normalization_24/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_24_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_24/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_24/FusedBatchNormV3FusedBatchNormV3dropout_39/Identity:output:0-batch_normalization_24/ReadVariableOp:value:0/batch_normalization_24/ReadVariableOp_1:value:0>batch_normalization_24/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
is_training( 2)
'batch_normalization_24/FusedBatchNormV3�
conv2d_54/Conv2D/ReadVariableOpReadVariableOp(conv2d_54_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_54/Conv2D/ReadVariableOp�
conv2d_54/Conv2DConv2D+batch_normalization_24/FusedBatchNormV3:y:0'conv2d_54/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@*
paddingVALID*
strides
2
conv2d_54/Conv2D�
 conv2d_54/BiasAdd/ReadVariableOpReadVariableOp)conv2d_54_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_54/BiasAdd/ReadVariableOp�
conv2d_54/BiasAddBiasAddconv2d_54/Conv2D:output:0(conv2d_54/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@2
conv2d_54/BiasAdd�
conv2d_54/SigmoidSigmoidconv2d_54/BiasAdd:output:0*
T0*/
_output_shapes
:���������		@2
conv2d_54/Sigmoid�
conv2d_54/mulMulconv2d_54/BiasAdd:output:0conv2d_54/Sigmoid:y:0*
T0*/
_output_shapes
:���������		@2
conv2d_54/mul�
conv2d_54/IdentityIdentityconv2d_54/mul:z:0*
T0*/
_output_shapes
:���������		@2
conv2d_54/Identity�
conv2d_54/IdentityN	IdentityNconv2d_54/mul:z:0conv2d_54/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-18034668*J
_output_shapes8
6:���������		@:���������		@2
conv2d_54/IdentityN�
conv2d_55/Conv2D/ReadVariableOpReadVariableOp(conv2d_55_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_55/Conv2D/ReadVariableOp�
conv2d_55/Conv2DConv2Dconv2d_54/IdentityN:output:0'conv2d_55/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
2
conv2d_55/Conv2D�
 conv2d_55/BiasAdd/ReadVariableOpReadVariableOp)conv2d_55_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_55/BiasAdd/ReadVariableOp�
conv2d_55/BiasAddBiasAddconv2d_55/Conv2D:output:0(conv2d_55/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_55/BiasAdd�
conv2d_55/SigmoidSigmoidconv2d_55/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
conv2d_55/Sigmoid�
conv2d_55/mulMulconv2d_55/BiasAdd:output:0conv2d_55/Sigmoid:y:0*
T0*/
_output_shapes
:���������@2
conv2d_55/mul�
conv2d_55/IdentityIdentityconv2d_55/mul:z:0*
T0*/
_output_shapes
:���������@2
conv2d_55/Identity�
conv2d_55/IdentityN	IdentityNconv2d_55/mul:z:0conv2d_55/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-18034680*J
_output_shapes8
6:���������@:���������@2
conv2d_55/IdentityN�
max_pooling2d_27/MaxPoolMaxPoolconv2d_55/IdentityN:output:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_27/MaxPool�
dropout_40/IdentityIdentity!max_pooling2d_27/MaxPool:output:0*
T0*/
_output_shapes
:���������@2
dropout_40/Identity�
%batch_normalization_25/ReadVariableOpReadVariableOp.batch_normalization_25_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_25/ReadVariableOp�
'batch_normalization_25/ReadVariableOp_1ReadVariableOp0batch_normalization_25_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_25/ReadVariableOp_1�
6batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_25/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_25/FusedBatchNormV3FusedBatchNormV3dropout_40/Identity:output:0-batch_normalization_25/ReadVariableOp:value:0/batch_normalization_25/ReadVariableOp_1:value:0>batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2)
'batch_normalization_25/FusedBatchNormV3u
flatten_13/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  2
flatten_13/Const�
flatten_13/ReshapeReshape+batch_normalization_25/FusedBatchNormV3:y:0flatten_13/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_13/Reshape�
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_26/MatMul/ReadVariableOp�
dense_26/MatMulMatMulflatten_13/Reshape:output:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_26/MatMul�
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_26/BiasAdd/ReadVariableOp�
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_26/BiasAdd}
dense_26/SigmoidSigmoiddense_26/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_26/Sigmoid�
dense_26/mulMuldense_26/BiasAdd:output:0dense_26/Sigmoid:y:0*
T0*(
_output_shapes
:����������2
dense_26/mulw
dense_26/IdentityIdentitydense_26/mul:z:0*
T0*(
_output_shapes
:����������2
dense_26/Identity�
dense_26/IdentityN	IdentityNdense_26/mul:z:0dense_26/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-18034710*<
_output_shapes*
(:����������:����������2
dense_26/IdentityN�
dropout_41/IdentityIdentitydense_26/IdentityN:output:0*
T0*(
_output_shapes
:����������2
dropout_41/Identity�
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes
:	�+*
dtype02 
dense_27/MatMul/ReadVariableOp�
dense_27/MatMulMatMuldropout_41/Identity:output:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������+2
dense_27/MatMul�
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:+*
dtype02!
dense_27/BiasAdd/ReadVariableOp�
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������+2
dense_27/BiasAdd|
dense_27/SoftmaxSoftmaxdense_27/BiasAdd:output:0*
T0*'
_output_shapes
:���������+2
dense_27/Softmaxn
IdentityIdentitydense_27/Softmax:softmax:0*
T0*'
_output_shapes
:���������+2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:���������:::::::::::::::::::::W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
j
N__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_18033448

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
9__inference_batch_normalization_25_layer_call_fn_18035118

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*]
fXRV
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_180337612
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�$
�
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_18033538

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
��
�

#__inference__wrapped_model_18033388
input_36:
6sequential_13_conv2d_52_conv2d_readvariableop_resource;
7sequential_13_conv2d_52_biasadd_readvariableop_resource:
6sequential_13_conv2d_53_conv2d_readvariableop_resource;
7sequential_13_conv2d_53_biasadd_readvariableop_resource@
<sequential_13_batch_normalization_24_readvariableop_resourceB
>sequential_13_batch_normalization_24_readvariableop_1_resourceQ
Msequential_13_batch_normalization_24_fusedbatchnormv3_readvariableop_resourceS
Osequential_13_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource:
6sequential_13_conv2d_54_conv2d_readvariableop_resource;
7sequential_13_conv2d_54_biasadd_readvariableop_resource:
6sequential_13_conv2d_55_conv2d_readvariableop_resource;
7sequential_13_conv2d_55_biasadd_readvariableop_resource@
<sequential_13_batch_normalization_25_readvariableop_resourceB
>sequential_13_batch_normalization_25_readvariableop_1_resourceQ
Msequential_13_batch_normalization_25_fusedbatchnormv3_readvariableop_resourceS
Osequential_13_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource9
5sequential_13_dense_26_matmul_readvariableop_resource:
6sequential_13_dense_26_biasadd_readvariableop_resource9
5sequential_13_dense_27_matmul_readvariableop_resource:
6sequential_13_dense_27_biasadd_readvariableop_resource
identity��
-sequential_13/conv2d_52/Conv2D/ReadVariableOpReadVariableOp6sequential_13_conv2d_52_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-sequential_13/conv2d_52/Conv2D/ReadVariableOp�
sequential_13/conv2d_52/Conv2DConv2Dinput_365sequential_13/conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2 
sequential_13/conv2d_52/Conv2D�
.sequential_13/conv2d_52/BiasAdd/ReadVariableOpReadVariableOp7sequential_13_conv2d_52_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_13/conv2d_52/BiasAdd/ReadVariableOp�
sequential_13/conv2d_52/BiasAddBiasAdd'sequential_13/conv2d_52/Conv2D:output:06sequential_13/conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2!
sequential_13/conv2d_52/BiasAdd�
sequential_13/conv2d_52/SigmoidSigmoid(sequential_13/conv2d_52/BiasAdd:output:0*
T0*/
_output_shapes
:��������� 2!
sequential_13/conv2d_52/Sigmoid�
sequential_13/conv2d_52/mulMul(sequential_13/conv2d_52/BiasAdd:output:0#sequential_13/conv2d_52/Sigmoid:y:0*
T0*/
_output_shapes
:��������� 2
sequential_13/conv2d_52/mul�
 sequential_13/conv2d_52/IdentityIdentitysequential_13/conv2d_52/mul:z:0*
T0*/
_output_shapes
:��������� 2"
 sequential_13/conv2d_52/Identity�
!sequential_13/conv2d_52/IdentityN	IdentityNsequential_13/conv2d_52/mul:z:0(sequential_13/conv2d_52/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-18033291*J
_output_shapes8
6:��������� :��������� 2#
!sequential_13/conv2d_52/IdentityN�
-sequential_13/conv2d_53/Conv2D/ReadVariableOpReadVariableOp6sequential_13_conv2d_53_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02/
-sequential_13/conv2d_53/Conv2D/ReadVariableOp�
sequential_13/conv2d_53/Conv2DConv2D*sequential_13/conv2d_52/IdentityN:output:05sequential_13/conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2 
sequential_13/conv2d_53/Conv2D�
.sequential_13/conv2d_53/BiasAdd/ReadVariableOpReadVariableOp7sequential_13_conv2d_53_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_13/conv2d_53/BiasAdd/ReadVariableOp�
sequential_13/conv2d_53/BiasAddBiasAdd'sequential_13/conv2d_53/Conv2D:output:06sequential_13/conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2!
sequential_13/conv2d_53/BiasAdd�
sequential_13/conv2d_53/SigmoidSigmoid(sequential_13/conv2d_53/BiasAdd:output:0*
T0*/
_output_shapes
:��������� 2!
sequential_13/conv2d_53/Sigmoid�
sequential_13/conv2d_53/mulMul(sequential_13/conv2d_53/BiasAdd:output:0#sequential_13/conv2d_53/Sigmoid:y:0*
T0*/
_output_shapes
:��������� 2
sequential_13/conv2d_53/mul�
 sequential_13/conv2d_53/IdentityIdentitysequential_13/conv2d_53/mul:z:0*
T0*/
_output_shapes
:��������� 2"
 sequential_13/conv2d_53/Identity�
!sequential_13/conv2d_53/IdentityN	IdentityNsequential_13/conv2d_53/mul:z:0(sequential_13/conv2d_53/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-18033303*J
_output_shapes8
6:��������� :��������� 2#
!sequential_13/conv2d_53/IdentityN�
&sequential_13/max_pooling2d_26/MaxPoolMaxPool*sequential_13/conv2d_53/IdentityN:output:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2(
&sequential_13/max_pooling2d_26/MaxPool�
!sequential_13/dropout_39/IdentityIdentity/sequential_13/max_pooling2d_26/MaxPool:output:0*
T0*/
_output_shapes
:��������� 2#
!sequential_13/dropout_39/Identity�
3sequential_13/batch_normalization_24/ReadVariableOpReadVariableOp<sequential_13_batch_normalization_24_readvariableop_resource*
_output_shapes
: *
dtype025
3sequential_13/batch_normalization_24/ReadVariableOp�
5sequential_13/batch_normalization_24/ReadVariableOp_1ReadVariableOp>sequential_13_batch_normalization_24_readvariableop_1_resource*
_output_shapes
: *
dtype027
5sequential_13/batch_normalization_24/ReadVariableOp_1�
Dsequential_13/batch_normalization_24/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_13_batch_normalization_24_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02F
Dsequential_13/batch_normalization_24/FusedBatchNormV3/ReadVariableOp�
Fsequential_13/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_13_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02H
Fsequential_13/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1�
5sequential_13/batch_normalization_24/FusedBatchNormV3FusedBatchNormV3*sequential_13/dropout_39/Identity:output:0;sequential_13/batch_normalization_24/ReadVariableOp:value:0=sequential_13/batch_normalization_24/ReadVariableOp_1:value:0Lsequential_13/batch_normalization_24/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_13/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
is_training( 27
5sequential_13/batch_normalization_24/FusedBatchNormV3�
-sequential_13/conv2d_54/Conv2D/ReadVariableOpReadVariableOp6sequential_13_conv2d_54_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02/
-sequential_13/conv2d_54/Conv2D/ReadVariableOp�
sequential_13/conv2d_54/Conv2DConv2D9sequential_13/batch_normalization_24/FusedBatchNormV3:y:05sequential_13/conv2d_54/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@*
paddingVALID*
strides
2 
sequential_13/conv2d_54/Conv2D�
.sequential_13/conv2d_54/BiasAdd/ReadVariableOpReadVariableOp7sequential_13_conv2d_54_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.sequential_13/conv2d_54/BiasAdd/ReadVariableOp�
sequential_13/conv2d_54/BiasAddBiasAdd'sequential_13/conv2d_54/Conv2D:output:06sequential_13/conv2d_54/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@2!
sequential_13/conv2d_54/BiasAdd�
sequential_13/conv2d_54/SigmoidSigmoid(sequential_13/conv2d_54/BiasAdd:output:0*
T0*/
_output_shapes
:���������		@2!
sequential_13/conv2d_54/Sigmoid�
sequential_13/conv2d_54/mulMul(sequential_13/conv2d_54/BiasAdd:output:0#sequential_13/conv2d_54/Sigmoid:y:0*
T0*/
_output_shapes
:���������		@2
sequential_13/conv2d_54/mul�
 sequential_13/conv2d_54/IdentityIdentitysequential_13/conv2d_54/mul:z:0*
T0*/
_output_shapes
:���������		@2"
 sequential_13/conv2d_54/Identity�
!sequential_13/conv2d_54/IdentityN	IdentityNsequential_13/conv2d_54/mul:z:0(sequential_13/conv2d_54/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-18033331*J
_output_shapes8
6:���������		@:���������		@2#
!sequential_13/conv2d_54/IdentityN�
-sequential_13/conv2d_55/Conv2D/ReadVariableOpReadVariableOp6sequential_13_conv2d_55_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02/
-sequential_13/conv2d_55/Conv2D/ReadVariableOp�
sequential_13/conv2d_55/Conv2DConv2D*sequential_13/conv2d_54/IdentityN:output:05sequential_13/conv2d_55/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
2 
sequential_13/conv2d_55/Conv2D�
.sequential_13/conv2d_55/BiasAdd/ReadVariableOpReadVariableOp7sequential_13_conv2d_55_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.sequential_13/conv2d_55/BiasAdd/ReadVariableOp�
sequential_13/conv2d_55/BiasAddBiasAdd'sequential_13/conv2d_55/Conv2D:output:06sequential_13/conv2d_55/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2!
sequential_13/conv2d_55/BiasAdd�
sequential_13/conv2d_55/SigmoidSigmoid(sequential_13/conv2d_55/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2!
sequential_13/conv2d_55/Sigmoid�
sequential_13/conv2d_55/mulMul(sequential_13/conv2d_55/BiasAdd:output:0#sequential_13/conv2d_55/Sigmoid:y:0*
T0*/
_output_shapes
:���������@2
sequential_13/conv2d_55/mul�
 sequential_13/conv2d_55/IdentityIdentitysequential_13/conv2d_55/mul:z:0*
T0*/
_output_shapes
:���������@2"
 sequential_13/conv2d_55/Identity�
!sequential_13/conv2d_55/IdentityN	IdentityNsequential_13/conv2d_55/mul:z:0(sequential_13/conv2d_55/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-18033343*J
_output_shapes8
6:���������@:���������@2#
!sequential_13/conv2d_55/IdentityN�
&sequential_13/max_pooling2d_27/MaxPoolMaxPool*sequential_13/conv2d_55/IdentityN:output:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2(
&sequential_13/max_pooling2d_27/MaxPool�
!sequential_13/dropout_40/IdentityIdentity/sequential_13/max_pooling2d_27/MaxPool:output:0*
T0*/
_output_shapes
:���������@2#
!sequential_13/dropout_40/Identity�
3sequential_13/batch_normalization_25/ReadVariableOpReadVariableOp<sequential_13_batch_normalization_25_readvariableop_resource*
_output_shapes
:@*
dtype025
3sequential_13/batch_normalization_25/ReadVariableOp�
5sequential_13/batch_normalization_25/ReadVariableOp_1ReadVariableOp>sequential_13_batch_normalization_25_readvariableop_1_resource*
_output_shapes
:@*
dtype027
5sequential_13/batch_normalization_25/ReadVariableOp_1�
Dsequential_13/batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_13_batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02F
Dsequential_13/batch_normalization_25/FusedBatchNormV3/ReadVariableOp�
Fsequential_13/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_13_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02H
Fsequential_13/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1�
5sequential_13/batch_normalization_25/FusedBatchNormV3FusedBatchNormV3*sequential_13/dropout_40/Identity:output:0;sequential_13/batch_normalization_25/ReadVariableOp:value:0=sequential_13/batch_normalization_25/ReadVariableOp_1:value:0Lsequential_13/batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_13/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 27
5sequential_13/batch_normalization_25/FusedBatchNormV3�
sequential_13/flatten_13/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  2 
sequential_13/flatten_13/Const�
 sequential_13/flatten_13/ReshapeReshape9sequential_13/batch_normalization_25/FusedBatchNormV3:y:0'sequential_13/flatten_13/Const:output:0*
T0*(
_output_shapes
:����������2"
 sequential_13/flatten_13/Reshape�
,sequential_13/dense_26/MatMul/ReadVariableOpReadVariableOp5sequential_13_dense_26_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02.
,sequential_13/dense_26/MatMul/ReadVariableOp�
sequential_13/dense_26/MatMulMatMul)sequential_13/flatten_13/Reshape:output:04sequential_13/dense_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_13/dense_26/MatMul�
-sequential_13/dense_26/BiasAdd/ReadVariableOpReadVariableOp6sequential_13_dense_26_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_13/dense_26/BiasAdd/ReadVariableOp�
sequential_13/dense_26/BiasAddBiasAdd'sequential_13/dense_26/MatMul:product:05sequential_13/dense_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2 
sequential_13/dense_26/BiasAdd�
sequential_13/dense_26/SigmoidSigmoid'sequential_13/dense_26/BiasAdd:output:0*
T0*(
_output_shapes
:����������2 
sequential_13/dense_26/Sigmoid�
sequential_13/dense_26/mulMul'sequential_13/dense_26/BiasAdd:output:0"sequential_13/dense_26/Sigmoid:y:0*
T0*(
_output_shapes
:����������2
sequential_13/dense_26/mul�
sequential_13/dense_26/IdentityIdentitysequential_13/dense_26/mul:z:0*
T0*(
_output_shapes
:����������2!
sequential_13/dense_26/Identity�
 sequential_13/dense_26/IdentityN	IdentityNsequential_13/dense_26/mul:z:0'sequential_13/dense_26/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-18033373*<
_output_shapes*
(:����������:����������2"
 sequential_13/dense_26/IdentityN�
!sequential_13/dropout_41/IdentityIdentity)sequential_13/dense_26/IdentityN:output:0*
T0*(
_output_shapes
:����������2#
!sequential_13/dropout_41/Identity�
,sequential_13/dense_27/MatMul/ReadVariableOpReadVariableOp5sequential_13_dense_27_matmul_readvariableop_resource*
_output_shapes
:	�+*
dtype02.
,sequential_13/dense_27/MatMul/ReadVariableOp�
sequential_13/dense_27/MatMulMatMul*sequential_13/dropout_41/Identity:output:04sequential_13/dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������+2
sequential_13/dense_27/MatMul�
-sequential_13/dense_27/BiasAdd/ReadVariableOpReadVariableOp6sequential_13_dense_27_biasadd_readvariableop_resource*
_output_shapes
:+*
dtype02/
-sequential_13/dense_27/BiasAdd/ReadVariableOp�
sequential_13/dense_27/BiasAddBiasAdd'sequential_13/dense_27/MatMul:product:05sequential_13/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������+2 
sequential_13/dense_27/BiasAdd�
sequential_13/dense_27/SoftmaxSoftmax'sequential_13/dense_27/BiasAdd:output:0*
T0*'
_output_shapes
:���������+2 
sequential_13/dense_27/Softmax|
IdentityIdentity(sequential_13/dense_27/Softmax:softmax:0*
T0*'
_output_shapes
:���������+2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:���������:::::::::::::::::::::Y U
/
_output_shapes
:���������
"
_user_specified_name
input_36:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
G__inference_conv2d_53_layer_call_and_return_conditional_losses_18033432

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource

identity_1��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� *
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� 2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+��������������������������� 2	
Sigmoid|
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*A
_output_shapes/
-:+��������������������������� 2
mulu
IdentityIdentitymul:z:0*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity�
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-18033425*n
_output_shapes\
Z:+��������������������������� :+��������������������������� 2
	IdentityN�

Identity_1IdentityIdentityN:output:0*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity_1"!

identity_1Identity_1:output:0*H
_input_shapes7
5:+��������������������������� :::i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
g
H__inference_dropout_39_layer_call_and_return_conditional_losses_18033799

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:��������� 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:��������� *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:��������� 2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:��������� 2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:��������� 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
F__inference_dense_26_layer_call_and_return_conditional_losses_18035220

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource

identity_1��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:����������2	
Sigmoidc
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*(
_output_shapes
:����������2
mul\
IdentityIdentitymul:z:0*
T0*(
_output_shapes
:����������2

Identity�
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-18035213*<
_output_shapes*
(:����������:����������2
	IdentityNk

Identity_1IdentityIdentityN:output:0*
T0*(
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
I
-__inference_dropout_41_layer_call_fn_18035256

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_dropout_41_layer_call_and_return_conditional_losses_180340892
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
&__inference_signature_wrapper_18034466
input_36
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_36unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*'
_output_shapes
:���������+*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*,
f'R%
#__inference__wrapped_model_180333882
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������+2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:���������::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:���������
"
_user_specified_name
input_36:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
0__inference_sequential_13_layer_call_fn_18034815

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*'
_output_shapes
:���������+*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_sequential_13_layer_call_and_return_conditional_losses_180343522
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������+2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:���������::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�D
�
K__inference_sequential_13_layer_call_and_return_conditional_losses_18034188
input_36
conv2d_52_18034133
conv2d_52_18034135
conv2d_53_18034138
conv2d_53_18034140#
batch_normalization_24_18034145#
batch_normalization_24_18034147#
batch_normalization_24_18034149#
batch_normalization_24_18034151
conv2d_54_18034154
conv2d_54_18034156
conv2d_55_18034159
conv2d_55_18034161#
batch_normalization_25_18034166#
batch_normalization_25_18034168#
batch_normalization_25_18034170#
batch_normalization_25_18034172
dense_26_18034176
dense_26_18034178
dense_27_18034182
dense_27_18034184
identity��.batch_normalization_24/StatefulPartitionedCall�.batch_normalization_25/StatefulPartitionedCall�!conv2d_52/StatefulPartitionedCall�!conv2d_53/StatefulPartitionedCall�!conv2d_54/StatefulPartitionedCall�!conv2d_55/StatefulPartitionedCall� dense_26/StatefulPartitionedCall� dense_27/StatefulPartitionedCall�
!conv2d_52/StatefulPartitionedCallStatefulPartitionedCallinput_36conv2d_52_18034133conv2d_52_18034135*
Tin
2*
Tout
2*/
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_conv2d_52_layer_call_and_return_conditional_losses_180334052#
!conv2d_52/StatefulPartitionedCall�
!conv2d_53/StatefulPartitionedCallStatefulPartitionedCall*conv2d_52/StatefulPartitionedCall:output:0conv2d_53_18034138conv2d_53_18034140*
Tin
2*
Tout
2*/
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_conv2d_53_layer_call_and_return_conditional_losses_180334322#
!conv2d_53/StatefulPartitionedCall�
 max_pooling2d_26/PartitionedCallPartitionedCall*conv2d_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_180334482"
 max_pooling2d_26/PartitionedCall�
dropout_39/PartitionedCallPartitionedCall)max_pooling2d_26/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_dropout_39_layer_call_and_return_conditional_losses_180338042
dropout_39/PartitionedCall�
.batch_normalization_24/StatefulPartitionedCallStatefulPartitionedCall#dropout_39/PartitionedCall:output:0batch_normalization_24_18034145batch_normalization_24_18034147batch_normalization_24_18034149batch_normalization_24_18034151*
Tin	
2*
Tout
2*/
_output_shapes
:��������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*]
fXRV
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_1803386520
.batch_normalization_24/StatefulPartitionedCall�
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_24/StatefulPartitionedCall:output:0conv2d_54_18034154conv2d_54_18034156*
Tin
2*
Tout
2*/
_output_shapes
:���������		@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_conv2d_54_layer_call_and_return_conditional_losses_180335972#
!conv2d_54/StatefulPartitionedCall�
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall*conv2d_54/StatefulPartitionedCall:output:0conv2d_55_18034159conv2d_55_18034161*
Tin
2*
Tout
2*/
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_conv2d_55_layer_call_and_return_conditional_losses_180336242#
!conv2d_55/StatefulPartitionedCall�
 max_pooling2d_27/PartitionedCallPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_max_pooling2d_27_layer_call_and_return_conditional_losses_180336402"
 max_pooling2d_27/PartitionedCall�
dropout_40/PartitionedCallPartitionedCall)max_pooling2d_27/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_dropout_40_layer_call_and_return_conditional_losses_180339292
dropout_40/PartitionedCall�
.batch_normalization_25/StatefulPartitionedCallStatefulPartitionedCall#dropout_40/PartitionedCall:output:0batch_normalization_25_18034166batch_normalization_25_18034168batch_normalization_25_18034170batch_normalization_25_18034172*
Tin	
2*
Tout
2*/
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*]
fXRV
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_1803399020
.batch_normalization_25/StatefulPartitionedCall�
flatten_13/PartitionedCallPartitionedCall7batch_normalization_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_flatten_13_layer_call_and_return_conditional_losses_180340322
flatten_13/PartitionedCall�
 dense_26/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_26_18034176dense_26_18034178*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dense_26_layer_call_and_return_conditional_losses_180340562"
 dense_26/StatefulPartitionedCall�
dropout_41/PartitionedCallPartitionedCall)dense_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_dropout_41_layer_call_and_return_conditional_losses_180340892
dropout_41/PartitionedCall�
 dense_27/StatefulPartitionedCallStatefulPartitionedCall#dropout_41/PartitionedCall:output:0dense_27_18034182dense_27_18034184*
Tin
2*
Tout
2*'
_output_shapes
:���������+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dense_27_layer_call_and_return_conditional_losses_180341132"
 dense_27/StatefulPartitionedCall�
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0/^batch_normalization_24/StatefulPartitionedCall/^batch_normalization_25/StatefulPartitionedCall"^conv2d_52/StatefulPartitionedCall"^conv2d_53/StatefulPartitionedCall"^conv2d_54/StatefulPartitionedCall"^conv2d_55/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall*
T0*'
_output_shapes
:���������+2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:���������::::::::::::::::::::2`
.batch_normalization_24/StatefulPartitionedCall.batch_normalization_24/StatefulPartitionedCall2`
.batch_normalization_25/StatefulPartitionedCall.batch_normalization_25/StatefulPartitionedCall2F
!conv2d_52/StatefulPartitionedCall!conv2d_52/StatefulPartitionedCall2F
!conv2d_53/StatefulPartitionedCall!conv2d_53/StatefulPartitionedCall2F
!conv2d_54/StatefulPartitionedCall!conv2d_54/StatefulPartitionedCall2F
!conv2d_55/StatefulPartitionedCall!conv2d_55/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall:Y U
/
_output_shapes
:���������
"
_user_specified_name
input_36:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
f
H__inference_dropout_40_layer_call_and_return_conditional_losses_18035021

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:���������@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
g
H__inference_dropout_41_layer_call_and_return_conditional_losses_18034084

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
0__inference_sequential_13_layer_call_fn_18034395
input_36
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_36unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*'
_output_shapes
:���������+*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_sequential_13_layer_call_and_return_conditional_losses_180343522
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������+2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:���������::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:���������
"
_user_specified_name
input_36:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_18035167

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@:::::W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�;
�	
!__inference__traced_save_18035363
file_prefix/
+savev2_conv2d_52_kernel_read_readvariableop-
)savev2_conv2d_52_bias_read_readvariableop/
+savev2_conv2d_53_kernel_read_readvariableop-
)savev2_conv2d_53_bias_read_readvariableop;
7savev2_batch_normalization_24_gamma_read_readvariableop:
6savev2_batch_normalization_24_beta_read_readvariableopA
=savev2_batch_normalization_24_moving_mean_read_readvariableopE
Asavev2_batch_normalization_24_moving_variance_read_readvariableop/
+savev2_conv2d_54_kernel_read_readvariableop-
)savev2_conv2d_54_bias_read_readvariableop/
+savev2_conv2d_55_kernel_read_readvariableop-
)savev2_conv2d_55_bias_read_readvariableop;
7savev2_batch_normalization_25_gamma_read_readvariableop:
6savev2_batch_normalization_25_beta_read_readvariableopA
=savev2_batch_normalization_25_moving_mean_read_readvariableopE
Asavev2_batch_normalization_25_moving_variance_read_readvariableop.
*savev2_dense_26_kernel_read_readvariableop,
(savev2_dense_26_bias_read_readvariableop.
*savev2_dense_27_kernel_read_readvariableop,
(savev2_dense_27_bias_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_49d005c244ef4ee8b03a17a0fb288141/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_52_kernel_read_readvariableop)savev2_conv2d_52_bias_read_readvariableop+savev2_conv2d_53_kernel_read_readvariableop)savev2_conv2d_53_bias_read_readvariableop7savev2_batch_normalization_24_gamma_read_readvariableop6savev2_batch_normalization_24_beta_read_readvariableop=savev2_batch_normalization_24_moving_mean_read_readvariableopAsavev2_batch_normalization_24_moving_variance_read_readvariableop+savev2_conv2d_54_kernel_read_readvariableop)savev2_conv2d_54_bias_read_readvariableop+savev2_conv2d_55_kernel_read_readvariableop)savev2_conv2d_55_bias_read_readvariableop7savev2_batch_normalization_25_gamma_read_readvariableop6savev2_batch_normalization_25_beta_read_readvariableop=savev2_batch_normalization_25_moving_mean_read_readvariableopAsavev2_batch_normalization_25_moving_variance_read_readvariableop*savev2_dense_26_kernel_read_readvariableop(savev2_dense_26_bias_read_readvariableop*savev2_dense_27_kernel_read_readvariableop(savev2_dense_27_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *"
dtypes
22
SaveV2�
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1�
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names�
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity�

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: : : :  : : : : : : @:@:@@:@:@:@:@:@:
��:�:	�+:+: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,	(
&
_output_shapes
: @: 


_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�+: 

_output_shapes
:+:

_output_shapes
: 
�
�
9__inference_batch_normalization_25_layer_call_fn_18035180

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*]
fXRV
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_180339722
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_18033761

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@:::::i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
f
H__inference_dropout_39_layer_call_and_return_conditional_losses_18033804

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:��������� 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:��������� 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
F__inference_dense_27_layer_call_and_return_conditional_losses_18035267

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�+*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������+2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:+*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������+2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������+2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:���������+2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_18035092

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@:::::i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
0__inference_sequential_13_layer_call_fn_18034292
input_36
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_36unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*'
_output_shapes
:���������+*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_sequential_13_layer_call_and_return_conditional_losses_180342492
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������+2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:���������::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:���������
"
_user_specified_name
input_36:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
f
H__inference_dropout_40_layer_call_and_return_conditional_losses_18033929

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:���������@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_18034978

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:��������� :::::W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_18033990

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@:::::W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
G__inference_conv2d_55_layer_call_and_return_conditional_losses_18033624

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource

identity_1��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@2	
Sigmoid|
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*A
_output_shapes/
-:+���������������������������@2
mulu
IdentityIdentitymul:z:0*
T0*A
_output_shapes/
-:+���������������������������@2

Identity�
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-18033617*n
_output_shapes\
Z:+���������������������������@:+���������������������������@2
	IdentityN�

Identity_1IdentityIdentityN:output:0*
T0*A
_output_shapes/
-:+���������������������������@2

Identity_1"!

identity_1Identity_1:output:0*H
_input_shapes7
5:+���������������������������@:::i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
O
3__inference_max_pooling2d_27_layer_call_fn_18033646

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_max_pooling2d_27_layer_call_and_return_conditional_losses_180336402
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�I
�
K__inference_sequential_13_layer_call_and_return_conditional_losses_18034249

inputs
conv2d_52_18034194
conv2d_52_18034196
conv2d_53_18034199
conv2d_53_18034201#
batch_normalization_24_18034206#
batch_normalization_24_18034208#
batch_normalization_24_18034210#
batch_normalization_24_18034212
conv2d_54_18034215
conv2d_54_18034217
conv2d_55_18034220
conv2d_55_18034222#
batch_normalization_25_18034227#
batch_normalization_25_18034229#
batch_normalization_25_18034231#
batch_normalization_25_18034233
dense_26_18034237
dense_26_18034239
dense_27_18034243
dense_27_18034245
identity��.batch_normalization_24/StatefulPartitionedCall�.batch_normalization_25/StatefulPartitionedCall�!conv2d_52/StatefulPartitionedCall�!conv2d_53/StatefulPartitionedCall�!conv2d_54/StatefulPartitionedCall�!conv2d_55/StatefulPartitionedCall� dense_26/StatefulPartitionedCall� dense_27/StatefulPartitionedCall�"dropout_39/StatefulPartitionedCall�"dropout_40/StatefulPartitionedCall�"dropout_41/StatefulPartitionedCall�
!conv2d_52/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_52_18034194conv2d_52_18034196*
Tin
2*
Tout
2*/
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_conv2d_52_layer_call_and_return_conditional_losses_180334052#
!conv2d_52/StatefulPartitionedCall�
!conv2d_53/StatefulPartitionedCallStatefulPartitionedCall*conv2d_52/StatefulPartitionedCall:output:0conv2d_53_18034199conv2d_53_18034201*
Tin
2*
Tout
2*/
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_conv2d_53_layer_call_and_return_conditional_losses_180334322#
!conv2d_53/StatefulPartitionedCall�
 max_pooling2d_26/PartitionedCallPartitionedCall*conv2d_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_180334482"
 max_pooling2d_26/PartitionedCall�
"dropout_39/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_26/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_dropout_39_layer_call_and_return_conditional_losses_180337992$
"dropout_39/StatefulPartitionedCall�
.batch_normalization_24/StatefulPartitionedCallStatefulPartitionedCall+dropout_39/StatefulPartitionedCall:output:0batch_normalization_24_18034206batch_normalization_24_18034208batch_normalization_24_18034210batch_normalization_24_18034212*
Tin	
2*
Tout
2*/
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*]
fXRV
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_1803384720
.batch_normalization_24/StatefulPartitionedCall�
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_24/StatefulPartitionedCall:output:0conv2d_54_18034215conv2d_54_18034217*
Tin
2*
Tout
2*/
_output_shapes
:���������		@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_conv2d_54_layer_call_and_return_conditional_losses_180335972#
!conv2d_54/StatefulPartitionedCall�
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall*conv2d_54/StatefulPartitionedCall:output:0conv2d_55_18034220conv2d_55_18034222*
Tin
2*
Tout
2*/
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_conv2d_55_layer_call_and_return_conditional_losses_180336242#
!conv2d_55/StatefulPartitionedCall�
 max_pooling2d_27/PartitionedCallPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_max_pooling2d_27_layer_call_and_return_conditional_losses_180336402"
 max_pooling2d_27/PartitionedCall�
"dropout_40/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_27/PartitionedCall:output:0#^dropout_39/StatefulPartitionedCall*
Tin
2*
Tout
2*/
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_dropout_40_layer_call_and_return_conditional_losses_180339242$
"dropout_40/StatefulPartitionedCall�
.batch_normalization_25/StatefulPartitionedCallStatefulPartitionedCall+dropout_40/StatefulPartitionedCall:output:0batch_normalization_25_18034227batch_normalization_25_18034229batch_normalization_25_18034231batch_normalization_25_18034233*
Tin	
2*
Tout
2*/
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*]
fXRV
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_1803397220
.batch_normalization_25/StatefulPartitionedCall�
flatten_13/PartitionedCallPartitionedCall7batch_normalization_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_flatten_13_layer_call_and_return_conditional_losses_180340322
flatten_13/PartitionedCall�
 dense_26/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_26_18034237dense_26_18034239*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dense_26_layer_call_and_return_conditional_losses_180340562"
 dense_26/StatefulPartitionedCall�
"dropout_41/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0#^dropout_40/StatefulPartitionedCall*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_dropout_41_layer_call_and_return_conditional_losses_180340842$
"dropout_41/StatefulPartitionedCall�
 dense_27/StatefulPartitionedCallStatefulPartitionedCall+dropout_41/StatefulPartitionedCall:output:0dense_27_18034243dense_27_18034245*
Tin
2*
Tout
2*'
_output_shapes
:���������+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dense_27_layer_call_and_return_conditional_losses_180341132"
 dense_27/StatefulPartitionedCall�
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0/^batch_normalization_24/StatefulPartitionedCall/^batch_normalization_25/StatefulPartitionedCall"^conv2d_52/StatefulPartitionedCall"^conv2d_53/StatefulPartitionedCall"^conv2d_54/StatefulPartitionedCall"^conv2d_55/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall#^dropout_39/StatefulPartitionedCall#^dropout_40/StatefulPartitionedCall#^dropout_41/StatefulPartitionedCall*
T0*'
_output_shapes
:���������+2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:���������::::::::::::::::::::2`
.batch_normalization_24/StatefulPartitionedCall.batch_normalization_24/StatefulPartitionedCall2`
.batch_normalization_25/StatefulPartitionedCall.batch_normalization_25/StatefulPartitionedCall2F
!conv2d_52/StatefulPartitionedCall!conv2d_52/StatefulPartitionedCall2F
!conv2d_53/StatefulPartitionedCall!conv2d_53/StatefulPartitionedCall2F
!conv2d_54/StatefulPartitionedCall!conv2d_54/StatefulPartitionedCall2F
!conv2d_55/StatefulPartitionedCall!conv2d_55/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2H
"dropout_39/StatefulPartitionedCall"dropout_39/StatefulPartitionedCall2H
"dropout_40/StatefulPartitionedCall"dropout_40/StatefulPartitionedCall2H
"dropout_41/StatefulPartitionedCall"dropout_41/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�$
�
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_18033730

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
d
H__inference_flatten_13_layer_call_and_return_conditional_losses_18035199

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�$
�
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_18034960

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:��������� ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�$
�
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_18034885

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
G__inference_conv2d_54_layer_call_and_return_conditional_losses_18033597

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource

identity_1��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@2	
Sigmoid|
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*A
_output_shapes/
-:+���������������������������@2
mulu
IdentityIdentitymul:z:0*
T0*A
_output_shapes/
-:+���������������������������@2

Identity�
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-18033590*n
_output_shapes\
Z:+���������������������������@:+���������������������������@2
	IdentityN�

Identity_1IdentityIdentityN:output:0*
T0*A
_output_shapes/
-:+���������������������������@2

Identity_1"!

identity_1Identity_1:output:0*H
_input_shapes7
5:+��������������������������� :::i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
+__inference_dense_26_layer_call_fn_18035229

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dense_26_layer_call_and_return_conditional_losses_180340562
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
g
H__inference_dropout_40_layer_call_and_return_conditional_losses_18035016

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������@2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������@2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
+__inference_dense_27_layer_call_fn_18035276

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:���������+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dense_27_layer_call_and_return_conditional_losses_180341132
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������+2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�D
�
K__inference_sequential_13_layer_call_and_return_conditional_losses_18034352

inputs
conv2d_52_18034297
conv2d_52_18034299
conv2d_53_18034302
conv2d_53_18034304#
batch_normalization_24_18034309#
batch_normalization_24_18034311#
batch_normalization_24_18034313#
batch_normalization_24_18034315
conv2d_54_18034318
conv2d_54_18034320
conv2d_55_18034323
conv2d_55_18034325#
batch_normalization_25_18034330#
batch_normalization_25_18034332#
batch_normalization_25_18034334#
batch_normalization_25_18034336
dense_26_18034340
dense_26_18034342
dense_27_18034346
dense_27_18034348
identity��.batch_normalization_24/StatefulPartitionedCall�.batch_normalization_25/StatefulPartitionedCall�!conv2d_52/StatefulPartitionedCall�!conv2d_53/StatefulPartitionedCall�!conv2d_54/StatefulPartitionedCall�!conv2d_55/StatefulPartitionedCall� dense_26/StatefulPartitionedCall� dense_27/StatefulPartitionedCall�
!conv2d_52/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_52_18034297conv2d_52_18034299*
Tin
2*
Tout
2*/
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_conv2d_52_layer_call_and_return_conditional_losses_180334052#
!conv2d_52/StatefulPartitionedCall�
!conv2d_53/StatefulPartitionedCallStatefulPartitionedCall*conv2d_52/StatefulPartitionedCall:output:0conv2d_53_18034302conv2d_53_18034304*
Tin
2*
Tout
2*/
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_conv2d_53_layer_call_and_return_conditional_losses_180334322#
!conv2d_53/StatefulPartitionedCall�
 max_pooling2d_26/PartitionedCallPartitionedCall*conv2d_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_180334482"
 max_pooling2d_26/PartitionedCall�
dropout_39/PartitionedCallPartitionedCall)max_pooling2d_26/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_dropout_39_layer_call_and_return_conditional_losses_180338042
dropout_39/PartitionedCall�
.batch_normalization_24/StatefulPartitionedCallStatefulPartitionedCall#dropout_39/PartitionedCall:output:0batch_normalization_24_18034309batch_normalization_24_18034311batch_normalization_24_18034313batch_normalization_24_18034315*
Tin	
2*
Tout
2*/
_output_shapes
:��������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*]
fXRV
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_1803386520
.batch_normalization_24/StatefulPartitionedCall�
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_24/StatefulPartitionedCall:output:0conv2d_54_18034318conv2d_54_18034320*
Tin
2*
Tout
2*/
_output_shapes
:���������		@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_conv2d_54_layer_call_and_return_conditional_losses_180335972#
!conv2d_54/StatefulPartitionedCall�
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall*conv2d_54/StatefulPartitionedCall:output:0conv2d_55_18034323conv2d_55_18034325*
Tin
2*
Tout
2*/
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_conv2d_55_layer_call_and_return_conditional_losses_180336242#
!conv2d_55/StatefulPartitionedCall�
 max_pooling2d_27/PartitionedCallPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_max_pooling2d_27_layer_call_and_return_conditional_losses_180336402"
 max_pooling2d_27/PartitionedCall�
dropout_40/PartitionedCallPartitionedCall)max_pooling2d_27/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_dropout_40_layer_call_and_return_conditional_losses_180339292
dropout_40/PartitionedCall�
.batch_normalization_25/StatefulPartitionedCallStatefulPartitionedCall#dropout_40/PartitionedCall:output:0batch_normalization_25_18034330batch_normalization_25_18034332batch_normalization_25_18034334batch_normalization_25_18034336*
Tin	
2*
Tout
2*/
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*]
fXRV
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_1803399020
.batch_normalization_25/StatefulPartitionedCall�
flatten_13/PartitionedCallPartitionedCall7batch_normalization_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_flatten_13_layer_call_and_return_conditional_losses_180340322
flatten_13/PartitionedCall�
 dense_26/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_26_18034340dense_26_18034342*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dense_26_layer_call_and_return_conditional_losses_180340562"
 dense_26/StatefulPartitionedCall�
dropout_41/PartitionedCallPartitionedCall)dense_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_dropout_41_layer_call_and_return_conditional_losses_180340892
dropout_41/PartitionedCall�
 dense_27/StatefulPartitionedCallStatefulPartitionedCall#dropout_41/PartitionedCall:output:0dense_27_18034346dense_27_18034348*
Tin
2*
Tout
2*'
_output_shapes
:���������+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dense_27_layer_call_and_return_conditional_losses_180341132"
 dense_27/StatefulPartitionedCall�
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0/^batch_normalization_24/StatefulPartitionedCall/^batch_normalization_25/StatefulPartitionedCall"^conv2d_52/StatefulPartitionedCall"^conv2d_53/StatefulPartitionedCall"^conv2d_54/StatefulPartitionedCall"^conv2d_55/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall*
T0*'
_output_shapes
:���������+2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:���������::::::::::::::::::::2`
.batch_normalization_24/StatefulPartitionedCall.batch_normalization_24/StatefulPartitionedCall2`
.batch_normalization_25/StatefulPartitionedCall.batch_normalization_25/StatefulPartitionedCall2F
!conv2d_52/StatefulPartitionedCall!conv2d_52/StatefulPartitionedCall2F
!conv2d_53/StatefulPartitionedCall!conv2d_53/StatefulPartitionedCall2F
!conv2d_54/StatefulPartitionedCall!conv2d_54/StatefulPartitionedCall2F
!conv2d_55/StatefulPartitionedCall!conv2d_55/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�$
�
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_18033847

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:��������� ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
��
�

K__inference_sequential_13_layer_call_and_return_conditional_losses_18034619

inputs,
(conv2d_52_conv2d_readvariableop_resource-
)conv2d_52_biasadd_readvariableop_resource,
(conv2d_53_conv2d_readvariableop_resource-
)conv2d_53_biasadd_readvariableop_resource2
.batch_normalization_24_readvariableop_resource4
0batch_normalization_24_readvariableop_1_resourceC
?batch_normalization_24_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_54_conv2d_readvariableop_resource-
)conv2d_54_biasadd_readvariableop_resource,
(conv2d_55_conv2d_readvariableop_resource-
)conv2d_55_biasadd_readvariableop_resource2
.batch_normalization_25_readvariableop_resource4
0batch_normalization_25_readvariableop_1_resourceC
?batch_normalization_25_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource+
'dense_26_matmul_readvariableop_resource,
(dense_26_biasadd_readvariableop_resource+
'dense_27_matmul_readvariableop_resource,
(dense_27_biasadd_readvariableop_resource
identity��:batch_normalization_24/AssignMovingAvg/AssignSubVariableOp�<batch_normalization_24/AssignMovingAvg_1/AssignSubVariableOp�:batch_normalization_25/AssignMovingAvg/AssignSubVariableOp�<batch_normalization_25/AssignMovingAvg_1/AssignSubVariableOp�
conv2d_52/Conv2D/ReadVariableOpReadVariableOp(conv2d_52_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_52/Conv2D/ReadVariableOp�
conv2d_52/Conv2DConv2Dinputs'conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
conv2d_52/Conv2D�
 conv2d_52/BiasAdd/ReadVariableOpReadVariableOp)conv2d_52_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_52/BiasAdd/ReadVariableOp�
conv2d_52/BiasAddBiasAddconv2d_52/Conv2D:output:0(conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
conv2d_52/BiasAdd�
conv2d_52/SigmoidSigmoidconv2d_52/BiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
conv2d_52/Sigmoid�
conv2d_52/mulMulconv2d_52/BiasAdd:output:0conv2d_52/Sigmoid:y:0*
T0*/
_output_shapes
:��������� 2
conv2d_52/mul�
conv2d_52/IdentityIdentityconv2d_52/mul:z:0*
T0*/
_output_shapes
:��������� 2
conv2d_52/Identity�
conv2d_52/IdentityN	IdentityNconv2d_52/mul:z:0conv2d_52/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-18034475*J
_output_shapes8
6:��������� :��������� 2
conv2d_52/IdentityN�
conv2d_53/Conv2D/ReadVariableOpReadVariableOp(conv2d_53_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_53/Conv2D/ReadVariableOp�
conv2d_53/Conv2DConv2Dconv2d_52/IdentityN:output:0'conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
conv2d_53/Conv2D�
 conv2d_53/BiasAdd/ReadVariableOpReadVariableOp)conv2d_53_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_53/BiasAdd/ReadVariableOp�
conv2d_53/BiasAddBiasAddconv2d_53/Conv2D:output:0(conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
conv2d_53/BiasAdd�
conv2d_53/SigmoidSigmoidconv2d_53/BiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
conv2d_53/Sigmoid�
conv2d_53/mulMulconv2d_53/BiasAdd:output:0conv2d_53/Sigmoid:y:0*
T0*/
_output_shapes
:��������� 2
conv2d_53/mul�
conv2d_53/IdentityIdentityconv2d_53/mul:z:0*
T0*/
_output_shapes
:��������� 2
conv2d_53/Identity�
conv2d_53/IdentityN	IdentityNconv2d_53/mul:z:0conv2d_53/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-18034487*J
_output_shapes8
6:��������� :��������� 2
conv2d_53/IdentityN�
max_pooling2d_26/MaxPoolMaxPoolconv2d_53/IdentityN:output:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2
max_pooling2d_26/MaxPooly
dropout_39/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout_39/dropout/Const�
dropout_39/dropout/MulMul!max_pooling2d_26/MaxPool:output:0!dropout_39/dropout/Const:output:0*
T0*/
_output_shapes
:��������� 2
dropout_39/dropout/Mul�
dropout_39/dropout/ShapeShape!max_pooling2d_26/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_39/dropout/Shape�
/dropout_39/dropout/random_uniform/RandomUniformRandomUniform!dropout_39/dropout/Shape:output:0*
T0*/
_output_shapes
:��������� *
dtype021
/dropout_39/dropout/random_uniform/RandomUniform�
!dropout_39/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2#
!dropout_39/dropout/GreaterEqual/y�
dropout_39/dropout/GreaterEqualGreaterEqual8dropout_39/dropout/random_uniform/RandomUniform:output:0*dropout_39/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:��������� 2!
dropout_39/dropout/GreaterEqual�
dropout_39/dropout/CastCast#dropout_39/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:��������� 2
dropout_39/dropout/Cast�
dropout_39/dropout/Mul_1Muldropout_39/dropout/Mul:z:0dropout_39/dropout/Cast:y:0*
T0*/
_output_shapes
:��������� 2
dropout_39/dropout/Mul_1�
%batch_normalization_24/ReadVariableOpReadVariableOp.batch_normalization_24_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_24/ReadVariableOp�
'batch_normalization_24/ReadVariableOp_1ReadVariableOp0batch_normalization_24_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_24/ReadVariableOp_1�
6batch_normalization_24/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_24_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_24/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_24/FusedBatchNormV3FusedBatchNormV3dropout_39/dropout/Mul_1:z:0-batch_normalization_24/ReadVariableOp:value:0/batch_normalization_24/ReadVariableOp_1:value:0>batch_normalization_24/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:2)
'batch_normalization_24/FusedBatchNormV3�
batch_normalization_24/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
batch_normalization_24/Const�
,batch_normalization_24/AssignMovingAvg/sub/xConst*R
_classH
FDloc:@batch_normalization_24/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2.
,batch_normalization_24/AssignMovingAvg/sub/x�
*batch_normalization_24/AssignMovingAvg/subSub5batch_normalization_24/AssignMovingAvg/sub/x:output:0%batch_normalization_24/Const:output:0*
T0*R
_classH
FDloc:@batch_normalization_24/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_24/AssignMovingAvg/sub�
5batch_normalization_24/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_24_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_24/AssignMovingAvg/ReadVariableOp�
,batch_normalization_24/AssignMovingAvg/sub_1Sub=batch_normalization_24/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_24/FusedBatchNormV3:batch_mean:0*
T0*R
_classH
FDloc:@batch_normalization_24/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2.
,batch_normalization_24/AssignMovingAvg/sub_1�
*batch_normalization_24/AssignMovingAvg/mulMul0batch_normalization_24/AssignMovingAvg/sub_1:z:0.batch_normalization_24/AssignMovingAvg/sub:z:0*
T0*R
_classH
FDloc:@batch_normalization_24/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_24/AssignMovingAvg/mul�
:batch_normalization_24/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp?batch_normalization_24_fusedbatchnormv3_readvariableop_resource.batch_normalization_24/AssignMovingAvg/mul:z:06^batch_normalization_24/AssignMovingAvg/ReadVariableOp7^batch_normalization_24/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_24/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02<
:batch_normalization_24/AssignMovingAvg/AssignSubVariableOp�
.batch_normalization_24/AssignMovingAvg_1/sub/xConst*T
_classJ
HFloc:@batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?20
.batch_normalization_24/AssignMovingAvg_1/sub/x�
,batch_normalization_24/AssignMovingAvg_1/subSub7batch_normalization_24/AssignMovingAvg_1/sub/x:output:0%batch_normalization_24/Const:output:0*
T0*T
_classJ
HFloc:@batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_24/AssignMovingAvg_1/sub�
7batch_normalization_24/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_24/AssignMovingAvg_1/ReadVariableOp�
.batch_normalization_24/AssignMovingAvg_1/sub_1Sub?batch_normalization_24/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_24/FusedBatchNormV3:batch_variance:0*
T0*T
_classJ
HFloc:@batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 20
.batch_normalization_24/AssignMovingAvg_1/sub_1�
,batch_normalization_24/AssignMovingAvg_1/mulMul2batch_normalization_24/AssignMovingAvg_1/sub_1:z:00batch_normalization_24/AssignMovingAvg_1/sub:z:0*
T0*T
_classJ
HFloc:@batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_24/AssignMovingAvg_1/mul�
<batch_normalization_24/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpAbatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource0batch_normalization_24/AssignMovingAvg_1/mul:z:08^batch_normalization_24/AssignMovingAvg_1/ReadVariableOp9^batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02>
<batch_normalization_24/AssignMovingAvg_1/AssignSubVariableOp�
conv2d_54/Conv2D/ReadVariableOpReadVariableOp(conv2d_54_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_54/Conv2D/ReadVariableOp�
conv2d_54/Conv2DConv2D+batch_normalization_24/FusedBatchNormV3:y:0'conv2d_54/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@*
paddingVALID*
strides
2
conv2d_54/Conv2D�
 conv2d_54/BiasAdd/ReadVariableOpReadVariableOp)conv2d_54_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_54/BiasAdd/ReadVariableOp�
conv2d_54/BiasAddBiasAddconv2d_54/Conv2D:output:0(conv2d_54/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������		@2
conv2d_54/BiasAdd�
conv2d_54/SigmoidSigmoidconv2d_54/BiasAdd:output:0*
T0*/
_output_shapes
:���������		@2
conv2d_54/Sigmoid�
conv2d_54/mulMulconv2d_54/BiasAdd:output:0conv2d_54/Sigmoid:y:0*
T0*/
_output_shapes
:���������		@2
conv2d_54/mul�
conv2d_54/IdentityIdentityconv2d_54/mul:z:0*
T0*/
_output_shapes
:���������		@2
conv2d_54/Identity�
conv2d_54/IdentityN	IdentityNconv2d_54/mul:z:0conv2d_54/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-18034535*J
_output_shapes8
6:���������		@:���������		@2
conv2d_54/IdentityN�
conv2d_55/Conv2D/ReadVariableOpReadVariableOp(conv2d_55_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_55/Conv2D/ReadVariableOp�
conv2d_55/Conv2DConv2Dconv2d_54/IdentityN:output:0'conv2d_55/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
2
conv2d_55/Conv2D�
 conv2d_55/BiasAdd/ReadVariableOpReadVariableOp)conv2d_55_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_55/BiasAdd/ReadVariableOp�
conv2d_55/BiasAddBiasAddconv2d_55/Conv2D:output:0(conv2d_55/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_55/BiasAdd�
conv2d_55/SigmoidSigmoidconv2d_55/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
conv2d_55/Sigmoid�
conv2d_55/mulMulconv2d_55/BiasAdd:output:0conv2d_55/Sigmoid:y:0*
T0*/
_output_shapes
:���������@2
conv2d_55/mul�
conv2d_55/IdentityIdentityconv2d_55/mul:z:0*
T0*/
_output_shapes
:���������@2
conv2d_55/Identity�
conv2d_55/IdentityN	IdentityNconv2d_55/mul:z:0conv2d_55/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-18034547*J
_output_shapes8
6:���������@:���������@2
conv2d_55/IdentityN�
max_pooling2d_27/MaxPoolMaxPoolconv2d_55/IdentityN:output:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_27/MaxPooly
dropout_40/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout_40/dropout/Const�
dropout_40/dropout/MulMul!max_pooling2d_27/MaxPool:output:0!dropout_40/dropout/Const:output:0*
T0*/
_output_shapes
:���������@2
dropout_40/dropout/Mul�
dropout_40/dropout/ShapeShape!max_pooling2d_27/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_40/dropout/Shape�
/dropout_40/dropout/random_uniform/RandomUniformRandomUniform!dropout_40/dropout/Shape:output:0*
T0*/
_output_shapes
:���������@*
dtype021
/dropout_40/dropout/random_uniform/RandomUniform�
!dropout_40/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2#
!dropout_40/dropout/GreaterEqual/y�
dropout_40/dropout/GreaterEqualGreaterEqual8dropout_40/dropout/random_uniform/RandomUniform:output:0*dropout_40/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������@2!
dropout_40/dropout/GreaterEqual�
dropout_40/dropout/CastCast#dropout_40/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������@2
dropout_40/dropout/Cast�
dropout_40/dropout/Mul_1Muldropout_40/dropout/Mul:z:0dropout_40/dropout/Cast:y:0*
T0*/
_output_shapes
:���������@2
dropout_40/dropout/Mul_1�
%batch_normalization_25/ReadVariableOpReadVariableOp.batch_normalization_25_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_25/ReadVariableOp�
'batch_normalization_25/ReadVariableOp_1ReadVariableOp0batch_normalization_25_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_25/ReadVariableOp_1�
6batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_25/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_25/FusedBatchNormV3FusedBatchNormV3dropout_40/dropout/Mul_1:z:0-batch_normalization_25/ReadVariableOp:value:0/batch_normalization_25/ReadVariableOp_1:value:0>batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2)
'batch_normalization_25/FusedBatchNormV3�
batch_normalization_25/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
batch_normalization_25/Const�
,batch_normalization_25/AssignMovingAvg/sub/xConst*R
_classH
FDloc:@batch_normalization_25/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2.
,batch_normalization_25/AssignMovingAvg/sub/x�
*batch_normalization_25/AssignMovingAvg/subSub5batch_normalization_25/AssignMovingAvg/sub/x:output:0%batch_normalization_25/Const:output:0*
T0*R
_classH
FDloc:@batch_normalization_25/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_25/AssignMovingAvg/sub�
5batch_normalization_25/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_25/AssignMovingAvg/ReadVariableOp�
,batch_normalization_25/AssignMovingAvg/sub_1Sub=batch_normalization_25/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_25/FusedBatchNormV3:batch_mean:0*
T0*R
_classH
FDloc:@batch_normalization_25/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2.
,batch_normalization_25/AssignMovingAvg/sub_1�
*batch_normalization_25/AssignMovingAvg/mulMul0batch_normalization_25/AssignMovingAvg/sub_1:z:0.batch_normalization_25/AssignMovingAvg/sub:z:0*
T0*R
_classH
FDloc:@batch_normalization_25/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2,
*batch_normalization_25/AssignMovingAvg/mul�
:batch_normalization_25/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp?batch_normalization_25_fusedbatchnormv3_readvariableop_resource.batch_normalization_25/AssignMovingAvg/mul:z:06^batch_normalization_25/AssignMovingAvg/ReadVariableOp7^batch_normalization_25/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_25/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02<
:batch_normalization_25/AssignMovingAvg/AssignSubVariableOp�
.batch_normalization_25/AssignMovingAvg_1/sub/xConst*T
_classJ
HFloc:@batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?20
.batch_normalization_25/AssignMovingAvg_1/sub/x�
,batch_normalization_25/AssignMovingAvg_1/subSub7batch_normalization_25/AssignMovingAvg_1/sub/x:output:0%batch_normalization_25/Const:output:0*
T0*T
_classJ
HFloc:@batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_25/AssignMovingAvg_1/sub�
7batch_normalization_25/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_25/AssignMovingAvg_1/ReadVariableOp�
.batch_normalization_25/AssignMovingAvg_1/sub_1Sub?batch_normalization_25/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_25/FusedBatchNormV3:batch_variance:0*
T0*T
_classJ
HFloc:@batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@20
.batch_normalization_25/AssignMovingAvg_1/sub_1�
,batch_normalization_25/AssignMovingAvg_1/mulMul2batch_normalization_25/AssignMovingAvg_1/sub_1:z:00batch_normalization_25/AssignMovingAvg_1/sub:z:0*
T0*T
_classJ
HFloc:@batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2.
,batch_normalization_25/AssignMovingAvg_1/mul�
<batch_normalization_25/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpAbatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource0batch_normalization_25/AssignMovingAvg_1/mul:z:08^batch_normalization_25/AssignMovingAvg_1/ReadVariableOp9^batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02>
<batch_normalization_25/AssignMovingAvg_1/AssignSubVariableOpu
flatten_13/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  2
flatten_13/Const�
flatten_13/ReshapeReshape+batch_normalization_25/FusedBatchNormV3:y:0flatten_13/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_13/Reshape�
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_26/MatMul/ReadVariableOp�
dense_26/MatMulMatMulflatten_13/Reshape:output:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_26/MatMul�
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_26/BiasAdd/ReadVariableOp�
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_26/BiasAdd}
dense_26/SigmoidSigmoiddense_26/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_26/Sigmoid�
dense_26/mulMuldense_26/BiasAdd:output:0dense_26/Sigmoid:y:0*
T0*(
_output_shapes
:����������2
dense_26/mulw
dense_26/IdentityIdentitydense_26/mul:z:0*
T0*(
_output_shapes
:����������2
dense_26/Identity�
dense_26/IdentityN	IdentityNdense_26/mul:z:0dense_26/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-18034597*<
_output_shapes*
(:����������:����������2
dense_26/IdentityNy
dropout_41/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout_41/dropout/Const�
dropout_41/dropout/MulMuldense_26/IdentityN:output:0!dropout_41/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_41/dropout/Mul
dropout_41/dropout/ShapeShapedense_26/IdentityN:output:0*
T0*
_output_shapes
:2
dropout_41/dropout/Shape�
/dropout_41/dropout/random_uniform/RandomUniformRandomUniform!dropout_41/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype021
/dropout_41/dropout/random_uniform/RandomUniform�
!dropout_41/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2#
!dropout_41/dropout/GreaterEqual/y�
dropout_41/dropout/GreaterEqualGreaterEqual8dropout_41/dropout/random_uniform/RandomUniform:output:0*dropout_41/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2!
dropout_41/dropout/GreaterEqual�
dropout_41/dropout/CastCast#dropout_41/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_41/dropout/Cast�
dropout_41/dropout/Mul_1Muldropout_41/dropout/Mul:z:0dropout_41/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_41/dropout/Mul_1�
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes
:	�+*
dtype02 
dense_27/MatMul/ReadVariableOp�
dense_27/MatMulMatMuldropout_41/dropout/Mul_1:z:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������+2
dense_27/MatMul�
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:+*
dtype02!
dense_27/BiasAdd/ReadVariableOp�
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������+2
dense_27/BiasAdd|
dense_27/SoftmaxSoftmaxdense_27/BiasAdd:output:0*
T0*'
_output_shapes
:���������+2
dense_27/Softmax�
IdentityIdentitydense_27/Softmax:softmax:0;^batch_normalization_24/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_24/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_25/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_25/AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:���������+2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:���������::::::::::::::::::::2x
:batch_normalization_24/AssignMovingAvg/AssignSubVariableOp:batch_normalization_24/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_24/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_24/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_25/AssignMovingAvg/AssignSubVariableOp:batch_normalization_25/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_25/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_25/AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
f
-__inference_dropout_39_layer_call_fn_18034837

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_dropout_39_layer_call_and_return_conditional_losses_180337992
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
g
H__inference_dropout_40_layer_call_and_return_conditional_losses_18033924

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������@2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������@2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
F__inference_dense_26_layer_call_and_return_conditional_losses_18034056

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource

identity_1��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:����������2	
Sigmoidc
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*(
_output_shapes
:����������2
mul\
IdentityIdentitymul:z:0*
T0*(
_output_shapes
:����������2

Identity�
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-18034049*<
_output_shapes*
(:����������:����������2
	IdentityNk

Identity_1IdentityIdentityN:output:0*
T0*(
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
9__inference_batch_normalization_24_layer_call_fn_18034929

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+��������������������������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*]
fXRV
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_180335692
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
f
H__inference_dropout_41_layer_call_and_return_conditional_losses_18034089

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:����������2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
f
H__inference_dropout_39_layer_call_and_return_conditional_losses_18034832

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:��������� 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:��������� 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�$
�
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_18035149

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
9__inference_batch_normalization_25_layer_call_fn_18035193

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*]
fXRV
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_180339902
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
j
N__inference_max_pooling2d_27_layer_call_and_return_conditional_losses_18033640

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
f
-__inference_dropout_41_layer_call_fn_18035251

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_dropout_41_layer_call_and_return_conditional_losses_180340842
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
,__inference_conv2d_53_layer_call_fn_18033442

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_conv2d_53_layer_call_and_return_conditional_losses_180334322
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
9__inference_batch_normalization_24_layer_call_fn_18035004

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:��������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*]
fXRV
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_180338652
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:��������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
,__inference_conv2d_54_layer_call_fn_18033607

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_conv2d_54_layer_call_and_return_conditional_losses_180335972
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_18034903

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� :::::i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
d
H__inference_flatten_13_layer_call_and_return_conditional_losses_18034032

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_18033865

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:��������� :::::W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
O
3__inference_max_pooling2d_26_layer_call_fn_18033454

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_180334482
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
f
-__inference_dropout_40_layer_call_fn_18035026

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_dropout_40_layer_call_and_return_conditional_losses_180339242
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
g
H__inference_dropout_39_layer_call_and_return_conditional_losses_18034827

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:��������� 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:��������� *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:��������� 2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:��������� 2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:��������� 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_18033569

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� :::::i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
f
H__inference_dropout_41_layer_call_and_return_conditional_losses_18035246

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:����������2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
I
-__inference_dropout_39_layer_call_fn_18034842

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_dropout_39_layer_call_and_return_conditional_losses_180338042
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�[
�
$__inference__traced_restore_18035435
file_prefix%
!assignvariableop_conv2d_52_kernel%
!assignvariableop_1_conv2d_52_bias'
#assignvariableop_2_conv2d_53_kernel%
!assignvariableop_3_conv2d_53_bias3
/assignvariableop_4_batch_normalization_24_gamma2
.assignvariableop_5_batch_normalization_24_beta9
5assignvariableop_6_batch_normalization_24_moving_mean=
9assignvariableop_7_batch_normalization_24_moving_variance'
#assignvariableop_8_conv2d_54_kernel%
!assignvariableop_9_conv2d_54_bias(
$assignvariableop_10_conv2d_55_kernel&
"assignvariableop_11_conv2d_55_bias4
0assignvariableop_12_batch_normalization_25_gamma3
/assignvariableop_13_batch_normalization_25_beta:
6assignvariableop_14_batch_normalization_25_moving_mean>
:assignvariableop_15_batch_normalization_25_moving_variance'
#assignvariableop_16_dense_26_kernel%
!assignvariableop_17_dense_26_bias'
#assignvariableop_18_dense_27_kernel%
!assignvariableop_19_dense_27_bias
identity_21��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_52_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_52_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_53_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_53_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp/assignvariableop_4_batch_normalization_24_gammaIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp.assignvariableop_5_batch_normalization_24_betaIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp5assignvariableop_6_batch_normalization_24_moving_meanIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp9assignvariableop_7_batch_normalization_24_moving_varianceIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_54_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_54_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv2d_55_kernelIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv2d_55_biasIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp0assignvariableop_12_batch_normalization_25_gammaIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp/assignvariableop_13_batch_normalization_25_betaIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp6assignvariableop_14_batch_normalization_25_moving_meanIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp:assignvariableop_15_batch_normalization_25_moving_varianceIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_26_kernelIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_26_biasIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp#assignvariableop_18_dense_27_kernelIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp!assignvariableop_19_dense_27_biasIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19�
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names�
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_20Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_20�
Identity_21IdentityIdentity_20:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_21"#
identity_21Identity_21:output:0*e
_input_shapesT
R: ::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
9__inference_batch_normalization_24_layer_call_fn_18034916

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*]
fXRV
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_180335382
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
,__inference_conv2d_52_layer_call_fn_18033415

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_conv2d_52_layer_call_and_return_conditional_losses_180334052
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
G__inference_conv2d_52_layer_call_and_return_conditional_losses_18033405

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource

identity_1��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� *
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� 2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+��������������������������� 2	
Sigmoid|
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*A
_output_shapes/
-:+��������������������������� 2
mulu
IdentityIdentitymul:z:0*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity�
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-18033398*n
_output_shapes\
Z:+��������������������������� :+��������������������������� 2
	IdentityN�

Identity_1IdentityIdentityN:output:0*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity_1"!

identity_1Identity_1:output:0*H
_input_shapes7
5:+���������������������������:::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�$
�
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_18033972

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
9__inference_batch_normalization_25_layer_call_fn_18035105

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*]
fXRV
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_180337302
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
E
input_369
serving_default_input_36:0���������<
dense_270
StatefulPartitionedCall:0���������+tensorflow/serving/predict:��
�i
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer-7
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer-12
layer_with_weights-7
layer-13
	optimizer

signatures
regularization_losses
trainable_variables
	variables
	keras_api
�_default_save_signature
+�&call_and_return_all_conditional_losses
�__call__"�d
_tf_keras_sequential�d{"class_name": "Sequential", "name": "sequential_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_13", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d_52", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 30, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_53", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_26", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_39", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_24", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_54", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_55", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_27", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_40", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_25", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_13", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_26", "trainable": true, "dtype": "float32", "units": 256, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_41", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 43, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 30, 3]}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 30, 3]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_13", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d_52", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 30, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_53", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_26", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_39", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_24", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_54", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_55", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_27", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_40", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_25", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_13", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_26", "trainable": true, "dtype": "float32", "units": 256, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_41", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 43, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 30, 3]}}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�


kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"class_name": "Conv2D", "name": "conv2d_52", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 30, 3]}, "stateful": false, "config": {"name": "conv2d_52", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 30, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 30, 3]}}
�	

kernel
bias
regularization_losses
trainable_variables
	variables
 	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_53", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_53", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 26, 32]}}
�
!regularization_losses
"trainable_variables
#	variables
$	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max_pooling2d_26", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
%regularization_losses
&trainable_variables
'	variables
(	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout_39", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_39", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
�	
)axis
	*gamma
+beta
,moving_mean
-moving_variance
.regularization_losses
/trainable_variables
0	variables
1	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_24", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_24", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 11, 11, 32]}}
�	

2kernel
3bias
4regularization_losses
5trainable_variables
6	variables
7	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_54", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_54", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 11, 11, 32]}}
�	

8kernel
9bias
:regularization_losses
;trainable_variables
<	variables
=	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_55", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_55", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 9, 64]}}
�
>regularization_losses
?trainable_variables
@	variables
A	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max_pooling2d_27", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout_40", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_40", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
�	
Faxis
	Ggamma
Hbeta
Imoving_mean
Jmoving_variance
Kregularization_losses
Ltrainable_variables
M	variables
N	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_25", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_25", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 3, 64]}}
�
Oregularization_losses
Ptrainable_variables
Q	variables
R	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "flatten_13", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�

Skernel
Tbias
Uregularization_losses
Vtrainable_variables
W	variables
X	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_26", "trainable": true, "dtype": "float32", "units": 256, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 576}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 576]}}
�
Yregularization_losses
Ztrainable_variables
[	variables
\	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout_41", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_41", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
�

]kernel
^bias
_regularization_losses
`trainable_variables
a	variables
b	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 43, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
"
	optimizer
-
�serving_default"
signature_map
 "
trackable_list_wrapper
�
0
1
2
3
*4
+5
26
37
88
99
G10
H11
S12
T13
]14
^15"
trackable_list_wrapper
�
0
1
2
3
*4
+5
,6
-7
28
39
810
911
G12
H13
I14
J15
S16
T17
]18
^19"
trackable_list_wrapper
�
regularization_losses
clayer_metrics
dmetrics
trainable_variables
enon_trainable_variables
flayer_regularization_losses
	variables

glayers
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:( 2conv2d_52/kernel
: 2conv2d_52/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
regularization_losses
hlayer_metrics
imetrics
trainable_variables
jnon_trainable_variables
klayer_regularization_losses
	variables

llayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_53/kernel
: 2conv2d_53/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
regularization_losses
mlayer_metrics
nmetrics
trainable_variables
onon_trainable_variables
player_regularization_losses
	variables

qlayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
!regularization_losses
rlayer_metrics
smetrics
"trainable_variables
tnon_trainable_variables
ulayer_regularization_losses
#	variables

vlayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
%regularization_losses
wlayer_metrics
xmetrics
&trainable_variables
ynon_trainable_variables
zlayer_regularization_losses
'	variables

{layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_24/gamma
):' 2batch_normalization_24/beta
2:0  (2"batch_normalization_24/moving_mean
6:4  (2&batch_normalization_24/moving_variance
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
<
*0
+1
,2
-3"
trackable_list_wrapper
�
.regularization_losses
|layer_metrics
}metrics
/trainable_variables
~non_trainable_variables
layer_regularization_losses
0	variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:( @2conv2d_54/kernel
:@2conv2d_54/bias
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
�
4regularization_losses
�layer_metrics
�metrics
5trainable_variables
�non_trainable_variables
 �layer_regularization_losses
6	variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(@@2conv2d_55/kernel
:@2conv2d_55/bias
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
�
:regularization_losses
�layer_metrics
�metrics
;trainable_variables
�non_trainable_variables
 �layer_regularization_losses
<	variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
>regularization_losses
�layer_metrics
�metrics
?trainable_variables
�non_trainable_variables
 �layer_regularization_losses
@	variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Bregularization_losses
�layer_metrics
�metrics
Ctrainable_variables
�non_trainable_variables
 �layer_regularization_losses
D	variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_25/gamma
):'@2batch_normalization_25/beta
2:0@ (2"batch_normalization_25/moving_mean
6:4@ (2&batch_normalization_25/moving_variance
 "
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
<
G0
H1
I2
J3"
trackable_list_wrapper
�
Kregularization_losses
�layer_metrics
�metrics
Ltrainable_variables
�non_trainable_variables
 �layer_regularization_losses
M	variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Oregularization_losses
�layer_metrics
�metrics
Ptrainable_variables
�non_trainable_variables
 �layer_regularization_losses
Q	variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
#:!
��2dense_26/kernel
:�2dense_26/bias
 "
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
�
Uregularization_losses
�layer_metrics
�metrics
Vtrainable_variables
�non_trainable_variables
 �layer_regularization_losses
W	variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Yregularization_losses
�layer_metrics
�metrics
Ztrainable_variables
�non_trainable_variables
 �layer_regularization_losses
[	variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
": 	�+2dense_27/kernel
:+2dense_27/bias
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
�
_regularization_losses
�layer_metrics
�metrics
`trainable_variables
�non_trainable_variables
 �layer_regularization_losses
a	variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<
,0
-1
I2
J3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
#__inference__wrapped_model_18033388�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� */�,
*�'
input_36���������
�2�
K__inference_sequential_13_layer_call_and_return_conditional_losses_18034725
K__inference_sequential_13_layer_call_and_return_conditional_losses_18034619
K__inference_sequential_13_layer_call_and_return_conditional_losses_18034188
K__inference_sequential_13_layer_call_and_return_conditional_losses_18034130�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
0__inference_sequential_13_layer_call_fn_18034395
0__inference_sequential_13_layer_call_fn_18034815
0__inference_sequential_13_layer_call_fn_18034770
0__inference_sequential_13_layer_call_fn_18034292�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
G__inference_conv2d_52_layer_call_and_return_conditional_losses_18033405�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
,__inference_conv2d_52_layer_call_fn_18033415�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
G__inference_conv2d_53_layer_call_and_return_conditional_losses_18033432�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
,__inference_conv2d_53_layer_call_fn_18033442�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
N__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_18033448�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
3__inference_max_pooling2d_26_layer_call_fn_18033454�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
H__inference_dropout_39_layer_call_and_return_conditional_losses_18034827
H__inference_dropout_39_layer_call_and_return_conditional_losses_18034832�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
-__inference_dropout_39_layer_call_fn_18034842
-__inference_dropout_39_layer_call_fn_18034837�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_18034885
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_18034978
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_18034903
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_18034960�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
9__inference_batch_normalization_24_layer_call_fn_18034929
9__inference_batch_normalization_24_layer_call_fn_18034991
9__inference_batch_normalization_24_layer_call_fn_18035004
9__inference_batch_normalization_24_layer_call_fn_18034916�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
G__inference_conv2d_54_layer_call_and_return_conditional_losses_18033597�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
,__inference_conv2d_54_layer_call_fn_18033607�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
G__inference_conv2d_55_layer_call_and_return_conditional_losses_18033624�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
,__inference_conv2d_55_layer_call_fn_18033634�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
N__inference_max_pooling2d_27_layer_call_and_return_conditional_losses_18033640�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
3__inference_max_pooling2d_27_layer_call_fn_18033646�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
H__inference_dropout_40_layer_call_and_return_conditional_losses_18035016
H__inference_dropout_40_layer_call_and_return_conditional_losses_18035021�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
-__inference_dropout_40_layer_call_fn_18035026
-__inference_dropout_40_layer_call_fn_18035031�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_18035074
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_18035092
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_18035167
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_18035149�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
9__inference_batch_normalization_25_layer_call_fn_18035193
9__inference_batch_normalization_25_layer_call_fn_18035118
9__inference_batch_normalization_25_layer_call_fn_18035180
9__inference_batch_normalization_25_layer_call_fn_18035105�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_flatten_13_layer_call_and_return_conditional_losses_18035199�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
-__inference_flatten_13_layer_call_fn_18035204�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_dense_26_layer_call_and_return_conditional_losses_18035220�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_dense_26_layer_call_fn_18035229�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_dropout_41_layer_call_and_return_conditional_losses_18035246
H__inference_dropout_41_layer_call_and_return_conditional_losses_18035241�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
-__inference_dropout_41_layer_call_fn_18035251
-__inference_dropout_41_layer_call_fn_18035256�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_dense_27_layer_call_and_return_conditional_losses_18035267�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_dense_27_layer_call_fn_18035276�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
6B4
&__inference_signature_wrapper_18034466input_36�
#__inference__wrapped_model_18033388�*+,-2389GHIJST]^9�6
/�,
*�'
input_36���������
� "3�0
.
dense_27"�
dense_27���������+�
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_18034885�*+,-M�J
C�@
:�7
inputs+��������������������������� 
p
� "?�<
5�2
0+��������������������������� 
� �
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_18034903�*+,-M�J
C�@
:�7
inputs+��������������������������� 
p 
� "?�<
5�2
0+��������������������������� 
� �
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_18034960r*+,-;�8
1�.
(�%
inputs��������� 
p
� "-�*
#� 
0��������� 
� �
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_18034978r*+,-;�8
1�.
(�%
inputs��������� 
p 
� "-�*
#� 
0��������� 
� �
9__inference_batch_normalization_24_layer_call_fn_18034916�*+,-M�J
C�@
:�7
inputs+��������������������������� 
p
� "2�/+��������������������������� �
9__inference_batch_normalization_24_layer_call_fn_18034929�*+,-M�J
C�@
:�7
inputs+��������������������������� 
p 
� "2�/+��������������������������� �
9__inference_batch_normalization_24_layer_call_fn_18034991e*+,-;�8
1�.
(�%
inputs��������� 
p
� " ���������� �
9__inference_batch_normalization_24_layer_call_fn_18035004e*+,-;�8
1�.
(�%
inputs��������� 
p 
� " ���������� �
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_18035074�GHIJM�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_18035092�GHIJM�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_18035149rGHIJ;�8
1�.
(�%
inputs���������@
p
� "-�*
#� 
0���������@
� �
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_18035167rGHIJ;�8
1�.
(�%
inputs���������@
p 
� "-�*
#� 
0���������@
� �
9__inference_batch_normalization_25_layer_call_fn_18035105�GHIJM�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
9__inference_batch_normalization_25_layer_call_fn_18035118�GHIJM�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
9__inference_batch_normalization_25_layer_call_fn_18035180eGHIJ;�8
1�.
(�%
inputs���������@
p
� " ����������@�
9__inference_batch_normalization_25_layer_call_fn_18035193eGHIJ;�8
1�.
(�%
inputs���������@
p 
� " ����������@�
G__inference_conv2d_52_layer_call_and_return_conditional_losses_18033405�I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+��������������������������� 
� �
,__inference_conv2d_52_layer_call_fn_18033415�I�F
?�<
:�7
inputs+���������������������������
� "2�/+��������������������������� �
G__inference_conv2d_53_layer_call_and_return_conditional_losses_18033432�I�F
?�<
:�7
inputs+��������������������������� 
� "?�<
5�2
0+��������������������������� 
� �
,__inference_conv2d_53_layer_call_fn_18033442�I�F
?�<
:�7
inputs+��������������������������� 
� "2�/+��������������������������� �
G__inference_conv2d_54_layer_call_and_return_conditional_losses_18033597�23I�F
?�<
:�7
inputs+��������������������������� 
� "?�<
5�2
0+���������������������������@
� �
,__inference_conv2d_54_layer_call_fn_18033607�23I�F
?�<
:�7
inputs+��������������������������� 
� "2�/+���������������������������@�
G__inference_conv2d_55_layer_call_and_return_conditional_losses_18033624�89I�F
?�<
:�7
inputs+���������������������������@
� "?�<
5�2
0+���������������������������@
� �
,__inference_conv2d_55_layer_call_fn_18033634�89I�F
?�<
:�7
inputs+���������������������������@
� "2�/+���������������������������@�
F__inference_dense_26_layer_call_and_return_conditional_losses_18035220^ST0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
+__inference_dense_26_layer_call_fn_18035229QST0�-
&�#
!�
inputs����������
� "������������
F__inference_dense_27_layer_call_and_return_conditional_losses_18035267]]^0�-
&�#
!�
inputs����������
� "%�"
�
0���������+
� 
+__inference_dense_27_layer_call_fn_18035276P]^0�-
&�#
!�
inputs����������
� "����������+�
H__inference_dropout_39_layer_call_and_return_conditional_losses_18034827l;�8
1�.
(�%
inputs��������� 
p
� "-�*
#� 
0��������� 
� �
H__inference_dropout_39_layer_call_and_return_conditional_losses_18034832l;�8
1�.
(�%
inputs��������� 
p 
� "-�*
#� 
0��������� 
� �
-__inference_dropout_39_layer_call_fn_18034837_;�8
1�.
(�%
inputs��������� 
p
� " ���������� �
-__inference_dropout_39_layer_call_fn_18034842_;�8
1�.
(�%
inputs��������� 
p 
� " ���������� �
H__inference_dropout_40_layer_call_and_return_conditional_losses_18035016l;�8
1�.
(�%
inputs���������@
p
� "-�*
#� 
0���������@
� �
H__inference_dropout_40_layer_call_and_return_conditional_losses_18035021l;�8
1�.
(�%
inputs���������@
p 
� "-�*
#� 
0���������@
� �
-__inference_dropout_40_layer_call_fn_18035026_;�8
1�.
(�%
inputs���������@
p
� " ����������@�
-__inference_dropout_40_layer_call_fn_18035031_;�8
1�.
(�%
inputs���������@
p 
� " ����������@�
H__inference_dropout_41_layer_call_and_return_conditional_losses_18035241^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
H__inference_dropout_41_layer_call_and_return_conditional_losses_18035246^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
-__inference_dropout_41_layer_call_fn_18035251Q4�1
*�'
!�
inputs����������
p
� "������������
-__inference_dropout_41_layer_call_fn_18035256Q4�1
*�'
!�
inputs����������
p 
� "������������
H__inference_flatten_13_layer_call_and_return_conditional_losses_18035199a7�4
-�*
(�%
inputs���������@
� "&�#
�
0����������
� �
-__inference_flatten_13_layer_call_fn_18035204T7�4
-�*
(�%
inputs���������@
� "������������
N__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_18033448�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
3__inference_max_pooling2d_26_layer_call_fn_18033454�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
N__inference_max_pooling2d_27_layer_call_and_return_conditional_losses_18033640�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
3__inference_max_pooling2d_27_layer_call_fn_18033646�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
K__inference_sequential_13_layer_call_and_return_conditional_losses_18034130�*+,-2389GHIJST]^A�>
7�4
*�'
input_36���������
p

 
� "%�"
�
0���������+
� �
K__inference_sequential_13_layer_call_and_return_conditional_losses_18034188�*+,-2389GHIJST]^A�>
7�4
*�'
input_36���������
p 

 
� "%�"
�
0���������+
� �
K__inference_sequential_13_layer_call_and_return_conditional_losses_18034619~*+,-2389GHIJST]^?�<
5�2
(�%
inputs���������
p

 
� "%�"
�
0���������+
� �
K__inference_sequential_13_layer_call_and_return_conditional_losses_18034725~*+,-2389GHIJST]^?�<
5�2
(�%
inputs���������
p 

 
� "%�"
�
0���������+
� �
0__inference_sequential_13_layer_call_fn_18034292s*+,-2389GHIJST]^A�>
7�4
*�'
input_36���������
p

 
� "����������+�
0__inference_sequential_13_layer_call_fn_18034395s*+,-2389GHIJST]^A�>
7�4
*�'
input_36���������
p 

 
� "����������+�
0__inference_sequential_13_layer_call_fn_18034770q*+,-2389GHIJST]^?�<
5�2
(�%
inputs���������
p

 
� "����������+�
0__inference_sequential_13_layer_call_fn_18034815q*+,-2389GHIJST]^?�<
5�2
(�%
inputs���������
p 

 
� "����������+�
&__inference_signature_wrapper_18034466�*+,-2389GHIJST]^E�B
� 
;�8
6
input_36*�'
input_36���������"3�0
.
dense_27"�
dense_27���������+