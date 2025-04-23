job "chexpert-diagnostic" {

  meta {
    owner = "sutariya"
  }

  datacenters = ["cluster"]
  type = "batch"

  # Ensure the job runs only on GPU nodes
  constraint {
    attribute = "${node.class}"
    value = "gpu"
  }

  group "simple-training-group" {

    # Define host volumes with proper read/write permissions
    volume "input" {
      type      = "host"
      read_only = true
      source    = "dl_input"
    }
    
    volume "output" {
      type      = "host"
      read_only = false
      source    = "dl_output"
    }

    volume "cache" {
      type      = "host"
      read_only = false
      source    = "dl_cache"
    }

    task "chexpert-diagnostic-training" {
      leader = true
      driver = "docker"
      config {
        image = "registry.fme.lan/dockerhub/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
        command = "bash"
        args = ["-c", "pip install pandas numpy matplotlib torchvision torch tqdm scikit-image wandb torchcontrib scikit-learn seaborn scikit-learn --root-user-action=ignore && export WANDB_API_KEY=c97efa068ce628aa2d4ad9bbc8b2b2dbaa6c6387 && wandb login && python chexpert_model_swa.py"]
        work_dir = "/deep_learning/output/Sutariya/chexpert"
        shm_size = 17179869184
        
      }

      # Mount the volumes correctly
      volume_mount {
        volume = "input"
        destination = "/deep_learning/input"
        read_only = true
      }
      
      volume_mount {
        volume = "output"
        destination = "/deep_learning/output"
        read_only = false
      }
      
      volume_mount {
        volume = "cache"
        destination = "/deep_learning/cache"
        read_only = false
      }      

      resources {
        cpu = 8000
        memory = 30000
        device "nvidia/gpu" {
          count = 1
        }
      }
    }

    restart {
      attempts = 2
      mode = "fail"
    }
    
    reschedule {
      attempts  = 0
      unlimited = false
    }
  }
}
