import { Injectable } from '@nestjs/common';
import { PrismaService } from 'src/prisma/prisma.service';

@Injectable()
export class ApointmentService {
  constructor(private readonly prismaService: PrismaService) {}

  create(data: any) {
    return this.prismaService.apointment.create({
      data: {
        name: data.name,
        contact: data.contact,
        dateTime: data.dateTime,
        doctorId: data.doctorId,
        userId: data.userId,
        status: 'PENDING',
      },
    });
  }

  findAllDoctorApointment(doctorId: string) {
    const dcId = parseInt(doctorId, 10);
    return this.prismaService.apointment.findMany({
      where: {
        doctorId: dcId,
      },
      include: {
        Doctor: true,
        User: true,
      },
    });
  }
  findAllUserApointment(userId: string) {
    const usId = parseInt(userId, 10);
    return this.prismaService.apointment.findMany({
      where: {
        userId: usId,
      },
      include: {
        Doctor: true,
        User: true,
      },
    });
  }

  changeStatus(data: any) {
    return this.prismaService.apointment.update({
      where: {
        id: data.apointmentId,
      },
      data: {
        status: data.status,
      },
    });
  }
}
