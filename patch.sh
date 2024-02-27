#!/usr/bin/env bash

SFTP_PATCH_SRC="patches/snakemake_storage_plugin_sftp_init.py"
SFTP_PATCH_DEST=$(python3 -c "import snakemake_storage_plugin_sftp as sftp; print(sftp.__file__)")
if [ ! -f "${SFTP_PATCH_DEST}.bak" ]; then
  mv "${SFTP_PATCH_DEST}" "${SFTP_PATCH_DEST}.bak"
fi

cp "${SFTP_PATCH_SRC}" "${SFTP_PATCH_DEST}"

