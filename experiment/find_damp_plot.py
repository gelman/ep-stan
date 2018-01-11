# load
K = 2
res_file = np.load('find_damp_K{}.npz'.format(K))
damps = res_file['damps']
mses = res_file['mses']
lls = res_file['lls']
kls = res_file['kls']
damps_selected = res_file['damps_selected']
lls_selected = res_file['lls_selected']
mses_selected = res_file['mses_selected']
kls_selected = res_file['kls_selected']
res_file.close()
iters, N_DAMP = kls.shape


# plot
plt.figure()
plt.plot(damps_selected)
plt.title('damps')

plt.figure()
plt.plot(mses_selected)
plt.title('mses')

plt.figure()
plt.plot(lls_selected)
plt.title('lls')

plt.figure()
plt.plot(kls_selected)
plt.title('kls')

fig, axes = plt.subplots(1, iters, sharex=True, sharey=True)
for i, ax in enumerate(axes):
    ax.plot(damps, mses[i], label=str(i+1))
fig.legend()
fig.suptitle('mses')

fig, axes = plt.subplots(1, iters, sharex=True, sharey=True)
for i, ax in enumerate(axes):
    ax.plot(damps, lls[i], label=str(i+1))
fig.legend()
fig.suptitle('lls')

fig, axes = plt.subplots(1, iters, sharex=True, sharey=True)
for i, ax in enumerate(axes):
    ax.plot(damps, kls[i], label=str(i+1))
fig.legend()
fig.suptitle('kls')
