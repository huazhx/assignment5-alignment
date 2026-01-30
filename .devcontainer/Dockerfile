FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

ARG USERNAME=cs336-user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# 修改顺序：先删用户，再删组
RUN apt-get update && apt-get install -y sudo \
    # [1. 先删用户] 检查占用了 UID 的用户并强制删除 (这一步会解除对组的占用)
    && if getent passwd $USER_UID; then userdel -r -f $(getent passwd $USER_UID | cut -d: -f1); fi \
    # [2. 再删组] 检查占用了 GID 的组并删除
    && if getent group $USER_GID; then groupdel $(getent group $USER_GID | cut -d: -f1); fi \
    # [3. 创建新用户]
    && groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# 修复 workspace 权限
RUN mkdir -p /workspace && chown -R $USERNAME:$USERNAME /workspace

ENV PATH="/home/$USERNAME/.local/bin:${PATH}"

USER $USERNAME