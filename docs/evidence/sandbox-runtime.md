# 本地 gVisor 前置条件核验

日期：2026-10-23。本记录是 Task 6 的环境准备，**没有通过代码 verifier 隔离验收**，没有执行模型生成的代码。

按 [gVisor 官方安装文档](https://gvisor.dev/docs/user_guide/install/) 使用完整 bundle，保留 gvisor-bin 辅助文件；仅放在本项目 `.tools/gvisor/20261019.0`。固定 [release-20261019.0](https://github.com/google/gvisor/releases/tag/release-20261019.0)，下载 73,621,897 字节，解压文件合计 164,899,549 字节，对照官方 SHA512 校验通过。URL、散列及文件清单见 [bundle.json](sandbox-runtime/bundle.json)。`runsc --version` 实际输出 release-20261019.0、spec 1.2.1。未安装到系统目录，未更改 Docker runtime、守护进程或系统策略。

安装前资源检查通过，项目预留增长 500 MB、下载 80 MB；实际占用仍低于十进制 14 GB，见 [安装后检查](sandbox-runtime/resource-post-install.json)。没有常驻服务、checkpoint、GPU 或新增费用。

实际主机入口执行两项可信探针：

- `unshare --user --map-root-user --mount --net /usr/bin/true`：exit 1，`write failed /proc/self/uid_map: Operation not permitted`。即使 unprivileged_userns_clone=1，当前入口仍不能据此认定 userns 可用。
- `runsc --root <项目内状态目录> --rootless --network=none --platform=systrap do --quiet /bin/echo crucible-gvisor-readiness`：exit 128，`re-executing self: fork/exec /proc/self/exe: permission denied`。完整参数与输出见 [rootless-smoke.json](sandbox-runtime/rootless-smoke.json)。只执行可信 echo 探针，未启动实际候选程序；未修改 AppArmor/sysctl 来绕过限制。

[探针后检查](sandbox-runtime/post-smoke.json) 没有发现本项目 runtime 进程或状态条目。失败原因尚未确诊，可能涉及当前执行入口权限；需要一个允许上述 rootless gVisor 启动的主机入口，或可配置的 Warden API，才能继续实际代码隔离验证。官方 [rootless 说明](https://gvisor.dev/docs/user_guide/rootless/) 描述了不同 user namespace 启动模式的限制。

后续只读核验确认本机 Docker 29.9.1/cgroup v2 可达，默认 runtime 为 runc，已配置 runtime 列表没有 gVisor，见 [精简记录](sandbox-runtime/docker-readiness.json)。这提供了将来在本项目范围设计独立封装的候选入口；本轮没有创建镜像、容器或改变 daemon，不代表 gVisor 已能运行。

2026-10-23 23:06（America/Los_Angeles）再次只读 Warden 公开 README：它已报告专用 HTTP→PG→scheduler→runc lab、节点退休协议及真实清理验收通过，取代此前“API→runc 接线待验”的状态。Crucible 未复跑或核对其原始证据。公开文档仍说明生产 HTTP 全部 401，生产构造器仅注册 runc；认证及 gVisor 服务前提未解除。没有读取 Warden 私有实现/凭证、借用 lab 或探测端口。

Task 6 仍可完成协议、适配器及 CPU 单元测试；实际沙箱不可用时应拒绝执行并给零奖励，不能用宿主机 subprocess 冒充沙箱。**未在真实多机上验证**。
