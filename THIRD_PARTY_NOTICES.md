# License scope and third-party notices

The root [MIT license](LICENSE) applies only to original contributions authored
for EnvPipe. It does not replace the licenses of third-party code or other
materials. Original EnvPipe additions to upstream files are offered under
MIT; the upstream portions and their applicable conditions remain unchanged.
Preserve the existing copyright, permission, and disclaimer notices when
redistributing those materials.

## DeepSpeed

`DeepSpeed/` includes Microsoft DeepSpeed code under the
[MIT license](DeepSpeed/LICENSE). Its existing license and file-level notices
are retained.

## Megatron-DeepSpeed

`Megatron-DeepSpeed/` includes NVIDIA Megatron code and other upstream code.
Its [LICENSE](Megatron-DeepSpeed/LICENSE) states BSD 3-Clause terms as the
default unless otherwise noted, and includes Apache-2.0 and MIT texts for
specified third-party portions. Consult that license and the individual
file notices rather than applying a single license to the entire directory.

## ResNet benchmark: upstream permission not established

`benchmark/single_node/model/resnet.py` references and incorporates code from
[weiaicunzai/pytorch-cifar100/models/resnet.py](https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/resnet.py).
As of October 30, 2026, the referenced upstream repository and source file
do not state a license, and no separate permission has been established in
this repository.

**This file is excluded from the root MIT license.** The source reference is
attribution, not a license grant. Permission from the upstream copyright
holder must be established, or the derived implementation replaced with
appropriately licensed code, before treating this file as freely reusable
or redistributable. The existing source reference is preserved.

## Separately obtained dependencies

Dependencies, pretrained models, and datasets obtained separately through the
setup instructions retain their respective licenses and terms. They are not
covered by EnvPipe's MIT grant.
