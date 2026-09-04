# MRI task variant of the lean proton runtime. All filesystem layers are shared
# with the lean CT image; this changes only task metadata.
ARG BASE_IMAGE=doserad-proton-ct:lean-runtime-20260830
FROM ${BASE_IMAGE}

ENV GC_TASK=proton_mri

LABEL org.grand-challenge.api-method="invoke"
