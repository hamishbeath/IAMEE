import env_sus
import econ_feas
import robust
import resources
import resilience

print('Running the framework')
env_sus.main()
print('Environmental sustainability done')
econ_feas.main()
print('Economic feasibility done')
robust.main()
print('Robustness done')
resources.main()
print('Resource efficiency done')
resilience.main()
print('Resilience done')

print('Framework run complete')